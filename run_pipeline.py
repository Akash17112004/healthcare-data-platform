#!/usr/bin/env python
"""Run the Smart Healthcare data pipeline.

    python run_pipeline.py                  # all stages, fresh data
    python run_pipeline.py --stage silver   # a single stage (needs the earlier ones to have run)

Stages: generate -> bronze -> silver -> process -> gold -> model
"""
import argparse
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from src import bronze, config, generate_sources, gold, model, processing, silver  # noqa: E402
from src.spark_session import get_spark  # noqa: E402

STAGES = ["generate", "bronze", "silver", "process", "gold", "model"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["all"] + STAGES, default="all")
    args = parser.parse_args()
    stages = STAGES if args.stage == "all" else [args.stage]

    if "generate" in stages:
        config.reset_workspace()
    config.FIGURES.mkdir(parents=True, exist_ok=True)

    spark = get_spark() if any(s != "generate" for s in stages) else None
    start = time.time()
    for stage in stages:
        t0 = time.time()
        print(f"\n========== stage: {stage} ==========")
        if stage == "generate":
            generate_sources.run()
        elif stage == "bronze":
            bronze.run(spark)
        elif stage == "silver":
            silver.run(spark)
        elif stage == "process":
            processing.run(spark)
        elif stage == "gold":
            gold.run(spark)
        elif stage == "model":
            model.run(spark)
        print(f"[{stage}] finished in {time.time() - t0:.1f}s")
    print(f"\nPipeline finished in {time.time() - start:.1f}s")
    if spark:
        spark.stop()


if __name__ == "__main__":
    main()
