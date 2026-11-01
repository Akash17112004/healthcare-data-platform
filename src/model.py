"""Stages 6-7 - Spark MLlib: train, compare, evaluate and apply a 30-day readmission classifier.

Task   : binary classification - will this patient be readmitted within 30 days of discharge?
Data   : data/gold/ml_dataset, rows where `in_model_population` is true (follow-up window complete, not deceased).
Split  : by PATIENT (hash of patient_id) -> 70% train / 15% validation / 15% test, so no patient is in two sets.
Models : Logistic Regression, Random Forest, Gradient-Boosted Trees (class-imbalance handled with row weights).
Select : best validation PR-AUC. The decision threshold is tuned on validation, never on test.
Output : data/predictions/* (Parquet), data/models/readmission (saved PipelineModel), reports/model_report.json, figures.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier  # noqa: E402
from pyspark.ml.evaluation import BinaryClassificationEvaluator  # noqa: E402
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from . import config as C  # noqa: E402
from .gold import FEATURE_CATEGORICAL, FEATURE_NUMERIC, LABEL  # noqa: E402
from .utils import to_pandas  # noqa: E402


def _stages(final_estimator, scale: bool):
    idx = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in FEATURE_CATEGORICAL]
    ohe = OneHotEncoder(inputCols=[f"{c}_idx" for c in FEATURE_CATEGORICAL],
                        outputCols=[f"{c}_oh" for c in FEATURE_CATEGORICAL], handleInvalid="keep")
    asm = VectorAssembler(inputCols=FEATURE_NUMERIC + [f"{c}_oh" for c in FEATURE_CATEGORICAL],
                          outputCol="features_raw", handleInvalid="keep")
    stages = idx + [ohe, asm]
    if scale:
        stages.append(StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True))
    else:
        stages[-1].setOutputCol("features")
    return stages + [final_estimator]


def _candidates():
    common = dict(featuresCol="features", labelCol="label", weightCol="weight")
    return {
        "LogisticRegression": Pipeline(stages=_stages(LogisticRegression(maxIter=100, regParam=0.01, elasticNetParam=0.0, **common), True)),
        "RandomForest": Pipeline(stages=_stages(RandomForestClassifier(numTrees=150, maxDepth=7, minInstancesPerNode=10, seed=C.SEED, **common), False)),
        "GradientBoostedTrees": Pipeline(stages=_stages(GBTClassifier(maxIter=60, maxDepth=3, stepSize=0.08, subsamplingRate=0.8, seed=C.SEED, **common), False)),
    }


def _feature_names(df):
    col = "features_raw" if "features_raw" in df.columns else "features"  # the scaler drops feature names
    attrs = df.schema[col].metadata["ml_attr"]["attrs"]
    return [a["name"] for kind in ("numeric", "binary", "nominal") for a in sorted(attrs.get(kind, []), key=lambda a: a["idx"])]


def _roc(y, p, points=60):
    order = np.argsort(-p)
    y = np.asarray(y)[order]
    tpr = np.cumsum(y) / max(y.sum(), 1)
    fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    keep = np.unique(np.linspace(0, len(y) - 1, points).astype(int))
    return np.r_[0, fpr[keep], 1], np.r_[0, tpr[keep], 1]


def _classification_metrics(y, p, thr):
    y = np.asarray(y)
    pred = (np.asarray(p) >= thr).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": round(float(thr), 4), "accuracy": round((tp + tn) / len(y), 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def run(spark) -> None:
    ml = spark.read.parquet(str(C.GOLD / "ml_dataset"))
    pop = (ml.filter("in_model_population")
             .withColumn("label", F.col(LABEL).cast("double"))
             .withColumn("bucket", F.pmod(F.hash("patient_id"), F.lit(100)))
             .withColumn("split", F.when(F.col("bucket") < 70, "train").when(F.col("bucket") < 85, "validation").otherwise("test")))
    train, val, test = (pop.filter(F.col("split") == s).cache() for s in ("train", "validation", "test"))
    n_tr, n_va, n_te = train.count(), val.count(), test.count()
    pos_rate = train.agg(F.avg("label")).first()[0]
    w_pos = (1 - pos_rate) / pos_rate
    weighted = lambda d: d.withColumn("weight", F.when(F.col("label") == 1, F.lit(w_pos)).otherwise(F.lit(1.0)))  # noqa: E731
    print(f"[model] population {n_tr + n_va + n_te:,} -> train {n_tr:,} / validation {n_va:,} / test {n_te:,} "
          f"(patient-level split) | train positive rate {100 * pos_rate:.1f}% -> positive weight {w_pos:.2f}")

    # ── compare candidates on the validation set ─────────────────────────────
    pr_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    roc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    fitted, rows = {}, []
    for name, pipe in _candidates().items():
        model = pipe.fit(weighted(train))
        pv = model.transform(val)
        fitted[name] = model
        rows.append({"model": name, "val_roc_auc": round(roc_eval.evaluate(pv), 4), "val_pr_auc": round(pr_eval.evaluate(pv), 4)})
        print(f"[model] {name:<22} validation ROC-AUC {rows[-1]['val_roc_auc']:.3f} | PR-AUC {rows[-1]['val_pr_auc']:.3f}")
    comparison = pd.DataFrame(rows)
    best_name = comparison.sort_values("val_pr_auc", ascending=False).iloc[0]["model"]
    best = fitted[best_name]
    print(f"[model] selected: {best_name}")

    def score(df):
        return (best.transform(df).withColumn("probability", vector_to_array("probability")[1])
                    .select("admission_id", "patient_id", "split", "label", "probability"))

    sv, st = to_pandas(score(val)), to_pandas(score(test))
    # threshold tuned on validation (max F1); never on test
    grid = np.linspace(0.05, 0.95, 91)
    thr = float(grid[int(np.argmax([_classification_metrics(sv["label"], sv["probability"], t)["f1"] for t in grid]))])
    test_cls = _classification_metrics(st["label"], st["probability"], thr)
    test_roc, test_pr = round(roc_eval.evaluate(best.transform(test)), 4), round(pr_eval.evaluate(best.transform(test)), 4)
    base_rate = float(st["label"].mean())

    # lift: how many true readmissions are in the 20% of patients the model ranks highest?
    st_sorted = st.sort_values("probability", ascending=False)
    top20 = st_sorted.head(int(round(0.2 * len(st_sorted))))
    capture_top20 = float(top20["label"].sum() / st["label"].sum())
    lift_top20 = float(top20["label"].mean() / base_rate)

    comparison["test_roc_auc"] = np.nan
    for name, model in fitted.items():
        comparison.loc[comparison["model"] == name, "test_roc_auc"] = round(roc_eval.evaluate(model.transform(test)), 4)

    # ── feature importance (tree models) or |coefficient| (logistic regression) ──
    final = best.stages[-1]
    names = _feature_names(best.transform(test.limit(5)))
    values = final.featureImportances.toArray() if hasattr(final, "featureImportances") else np.abs(final.coefficients.toArray())
    imp = (pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False))
    imp["importance"] = (imp["importance"] / imp["importance"].sum()).round(4)

    # ── score every admission (also non-modelled ones) -> predictions table ──
    allp = (best.transform(ml.withColumn("label", F.col(LABEL).cast("double")))
              .withColumn("probability", F.round(vector_to_array("probability")[1], 4)))
    split_lookup = pop.select("admission_id", "split")
    cuts = np.quantile(sv["probability"], [0.5, 0.8])  # validation-based bands: top 20% High, next 30% Medium
    predictions = (allp.join(split_lookup, "admission_id", "left")
        .withColumn("split", F.coalesce("split", F.lit("not_modelled")))
        .withColumn("predicted_readmit", (F.col("probability") >= thr).cast("int"))
        .withColumn("risk_band", F.when(F.col("probability") >= float(cuts[1]), "High")
                                  .when(F.col("probability") >= float(cuts[0]), "Medium").otherwise("Low"))
        .select("admission_id", "patient_id", "department", "admission_type", "age", "gender", "severity", "admit_ts",
                "discharge_ts", "outcome", F.col(LABEL).alias("actual_readmitted_30d"), "split",
                "probability", "predicted_readmit", "risk_band"))
    predictions.write.mode("overwrite").parquet(str(C.PREDICTIONS / "readmission_predictions"))

    # ── small result tables as Parquet + JSON report ────────────────────────
    fpr, tpr = _roc(st["label"], st["probability"])
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_parquet(C.PREDICTIONS / "roc_curve.parquet", index=False)
    comparison.to_parquet(C.PREDICTIONS / "model_comparison.parquet", index=False)
    imp.to_parquet(C.PREDICTIONS / "feature_importance.parquet", index=False)
    report = {"selected_model": best_name, "split_rows": {"train": n_tr, "validation": n_va, "test": n_te},
              "test_positive_rate": round(base_rate, 4), "test_roc_auc": test_roc, "test_pr_auc": test_pr,
              "test_pr_auc_baseline": round(base_rate, 4), "test_at_threshold": test_cls,
              "top20pct_capture_of_readmissions": round(capture_top20, 4), "top20pct_lift": round(lift_top20, 3),
              "risk_band_cutoffs": {"medium_from": round(float(cuts[0]), 4), "high_from": round(float(cuts[1]), 4)},
              "label_note": "Simulated data: model quality reflects the signal injected by the generator, not real clinical performance."}
    (C.REPORTS / "model_report.json").write_text(json.dumps(report, indent=2))
    comparison.to_csv(C.REPORTS / "model_comparison.csv", index=False)
    imp.head(15).to_csv(C.REPORTS / "feature_importance.csv", index=False)
    best.write().overwrite().save(str(C.MODELS / "readmission_pipeline"))

    _figures(fpr, tpr, test_roc, imp, comparison, best_name)
    print(f"[model] TEST  ROC-AUC {test_roc:.3f} | PR-AUC {test_pr:.3f} (baseline {base_rate:.3f}) | "
          f"threshold {thr:.2f}: precision {test_cls['precision']:.3f} recall {test_cls['recall']:.3f} F1 {test_cls['f1']:.3f}")
    print(f"[model] top-20% highest-risk patients contain {100 * capture_top20:.1f}% of all readmissions (lift {lift_top20:.2f}x)")
    print("[model] top features:", ", ".join(imp.head(6)["feature"]))
    print(f"[model] saved predictions ({predictions.count():,} rows), metrics, importance and the fitted pipeline")


def _figures(fpr, tpr, auc, imp, comparison, best_name):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(fpr, tpr, color="#1f6f8b", lw=2, label=f"{best_name} (AUC {auc:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", label="Random guess")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve on the held-out test patients")
    ax[0].legend(loc="lower right", frameon=False)
    top = imp.head(10).iloc[::-1]
    ax[1].barh(top["feature"], top["importance"], color="#1f6f8b")
    ax[1].set(xlabel="Share of total importance", title="What the model relies on most")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "model_performance.png", dpi=130)
    plt.close(fig)
