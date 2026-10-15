"""Stage 1 - Source systems (simulated).

Real hospital data cannot be shared, so the three source systems are generated here with a fixed
seed. Realistic data-quality problems are injected on purpose (duplicates, nulls, numbers stored as
text, inconsistent spellings, mixed units/formats, impossible values, orphan records) so that the
Silver stage has genuine work to do.

  * patients_registry.csv   - CSV export from the registration desk
  * hospital_his.db         - relational database (SQLite stand-in), table `admissions`
  * vitals_api/*.json       - paginated REST API pages with bedside readings
"""
import json
import random
import sqlite3
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

from . import config as C

CITIES = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune", "Jaipur", "Lucknow", "Chandigarh"]
BLOOD = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
INSUR = ["Government", "Private", "Corporate", "Uninsured"]
BOOL_STYLES = [("Y", "N"), ("Yes", "No"), ("1", "0"), ("True", "False"), ("y", "n")]
G_VARIANTS = {"Male": ["Male", "male", "M", "MALE", "m"], "Female": ["Female", "female", "F", "FEMALE", "f"]}

DEPTS = {  # dept: (diagnoses, typical length of stay in days, cost per day in INR)
    "Cardiology": (["Heart Failure", "Myocardial Infarction", "Hypertensive Crisis", "Arrhythmia"], 5, 14000),
    "Endocrinology": (["Diabetic Ketoacidosis", "Type 2 Diabetes", "Thyroid Disorder"], 4, 9000),
    "Pulmonology": (["COPD Exacerbation", "Pneumonia", "Asthma"], 5, 11000),
    "Orthopedics": (["Fracture", "Joint Replacement", "Spinal Injury"], 6, 16000),
    "Neurology": (["Stroke", "Epilepsy", "Migraine"], 6, 15000),
    "General Medicine": (["Viral Fever", "Dengue", "Typhoid", "Gastroenteritis"], 3, 6000),
    "Nephrology": (["Chronic Kidney Disease", "Kidney Stones"], 5, 12000),
}
DEPT_W = [.16, .12, .13, .14, .12, .25, .08]
ABBR = {"Cardiology": "Cardio", "Orthopedics": "Ortho", "General Medicine": "Gen Med", "Neurology": "Neuro"}
SEV_WORDS = {1: "low", 2: "mild", 3: "moderate", 4: "high", 5: "critical"}
DEVICES = ["Bedside Monitor", "Wearable", "Manual Entry"]
IST = timezone(timedelta(hours=5, minutes=30))


# ───────────────────────── patients (CSV) ─────────────────────────
def _messy_case(s, p_upper=0.07, p_lower=0.05, p_pad=0.04):
    r = random.random()
    if r < p_upper:
        return s.upper()
    if r < p_upper + p_lower:
        return s.lower()
    if r < p_upper + p_lower + p_pad:
        return f"  {s}  "
    return s


def _enc_bool(v):
    if random.random() < 0.03:
        return None
    yes, no = random.choice(BOOL_STYLES)
    return yes if v else no


def _gen_patients():
    patients, truth = [], {}
    for i in range(1, C.N_PATIENTS + 1):
        pid = f"P{i:05d}"
        age = int(np.clip(np.random.gamma(4.5, 10), 1, 92))
        dob = C.REF_DATE - timedelta(days=age * 365 + random.randint(0, 364))
        dia = random.random() < 0.03 + age / 300
        htn = random.random() < 0.05 + age / 200
        gender = random.choices(["Male", "Female", "Other"], [0.49, 0.49, 0.02])[0]
        truth[pid] = dict(age=age, dia=dia, htn=htn)  # hidden ground truth that keeps later data realistic

        r = random.random()
        if r < 0.03:
            dob_s = None
        elif r < 0.04:
            dob_s = (C.REF_DATE + timedelta(days=random.randint(30, 900))).strftime("%Y-%m-%d")  # DOB in the future
        elif r < 0.05:
            dob_s = "1890-01-01"  # impossible age
        elif r < 0.11:
            dob_s = dob.strftime("%d/%m/%Y")  # different date format
        else:
            dob_s = dob.strftime("%Y-%m-%d")

        g_s = None if random.random() < 0.04 else (
            random.choice(G_VARIANTS[gender]) if gender in G_VARIANTS else random.choice(["Other", "other"]))
        city = None if random.random() < 0.06 else _messy_case(random.choice(CITIES))
        bg = random.choices(BLOOD, [.30, .06, .25, .04, .25, .04, .04, .02])[0]
        r = random.random()
        bg = None if r < .05 else bg.lower() if r < .08 else "Unknown" if r < .10 else bg
        ins = None if random.random() < 0.05 else _messy_case(random.choices(INSUR, [.35, .30, .20, .15])[0], .10, .05, .03)
        reg = (C.START - timedelta(days=random.randint(0, 1500))).strftime("%Y-%m-%d")
        upd = C.REF_DATE - timedelta(days=random.randint(0, 365), hours=random.randint(0, 23))
        patients.append(dict(patient_id=pid, gender=g_s, dob=dob_s, city=city, blood_group=bg, insurance_type=ins,
                             has_diabetes=_enc_bool(dia), has_hypertension=_enc_bool(htn),
                             registration_date=reg, last_updated=upd.strftime("%Y-%m-%d %H:%M:%S")))

    # registry re-sync glitch: ~4% of patients appear twice, the newer copy has a different city
    dups = []
    for rec in random.sample(patients, int(0.04 * len(patients))):
        d = dict(rec)
        d["city"] = random.choice(CITIES)
        d["last_updated"] = (pd.Timestamp(rec["last_updated"]) + timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d %H:%M:%S")
        dups.append(d)
    for _ in range(6):  # a few rows with no patient_id at all
        d = dict(random.choice(patients))
        d["patient_id"] = None
        dups.append(d)

    df = pd.DataFrame(patients + dups).sample(frac=1, random_state=1).reset_index(drop=True)
    df.to_csv(C.RAW / "patients_registry.csv", index=False)
    return truth, len(df)


# ───────────────────────── admissions (database) ─────────────────────────
def _make_adm(truth, pids, pid, admit, dept=None, atype=None):
    dept = dept or random.choices(list(DEPTS), DEPT_W)[0]
    diags, base_los, cpd = DEPTS[dept]
    sev = random.choices([1, 2, 3, 4, 5], [.15, .30, .30, .18, .07])[0]
    atype = atype or random.choices(["Emergency", "Elective", "Urgent"], [.45, .35, .20])[0]
    los = max(1, int(round(base_los * (0.6 + 0.25 * sev) * np.random.lognormal(0, 0.35))))
    disch = admit + timedelta(days=los, hours=random.randint(-3, 5))
    bill = cpd * los * (0.8 + 0.15 * sev) * np.random.lognormal(0, 0.25) * (1.15 if atype == "Emergency" else 1)
    r = random.random()
    p_death = 0.0015 * sev ** 2
    outcome = "Deceased" if r < p_death else "Transferred" if r < p_death + 0.03 else "Discharged"
    return dict(patient_id=pid, department=dept,
                doctor_id=f"D{list(DEPTS).index(dept) * 10 + random.randint(1, 9):03d}",
                admission_type=atype, primary_diagnosis=random.choice(diags), severity=sev,
                admit=admit, discharge=disch, los=los, bill=int(round(bill)), outcome=outcome)


def _fmt_dt(dt):
    return dt.strftime("%d/%m/%Y %H:%M") if random.random() < 0.10 else dt.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_bill(b):
    r = random.random()
    if r < .04:
        return None
    if r < .06:
        return str(-abs(b))  # invalid negative bill
    if r < .18:
        return f"₹{b:,}"
    if r < .26:
        return f"{b} INR"
    if r < .30:
        return f"Rs. {b:,}"
    return str(b)


def _messy_dept(d):
    r = random.random()
    if r < .05:
        return d.upper()
    if r < .08:
        return d.lower()
    if r < .11:
        return f"  {d} "
    if r < .13 and d in ABBR:
        return ABBR[d]
    return d


def _messy_type(t):
    r = random.random()
    return t.lower() if r < .06 else t.upper() if r < .10 else "ER" if (t == "Emergency" and r < .13) else t


def _messy_outcome(o):
    r = random.random()
    return o.lower() if r < .05 else o.upper() if r < .08 else "Recovered" if (o == "Discharged" and r < .12) else o


def _gen_admissions(truth):
    pids = list(truth)
    p_weights = [1 + 2 * t["dia"] + 2 * t["htn"] + t["age"] / 40 for t in truth.values()]  # chronic/older -> more admissions

    base = []
    for _ in range(5200):
        admit = C.START + timedelta(days=random.randint(0, 715), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        base.append(_make_adm(truth, pids, random.choices(pids, p_weights)[0], admit))

    follow_ups = []  # ~14% of admissions are followed by a re-admission within 2-28 days
    for a in base:
        if a["outcome"] != "Deceased" and random.random() < 0.14:
            nxt = a["discharge"] + timedelta(days=random.randint(2, 28), hours=random.randint(0, 23))
            if nxt < C.REF_DATE - timedelta(days=10):
                follow_ups.append(_make_adm(truth, pids, a["patient_id"], nxt,
                                            dept=a["department"] if random.random() < .7 else None, atype="Emergency"))

    adm_true = sorted(base + follow_ups, key=lambda a: a["admit"])
    for n, a in enumerate(adm_true, 1):
        a["admission_id"] = f"A{n:06d}"

    rows = []
    for a in adm_true:
        r = random.random()
        pid = a["patient_id"] if r > .015 else (f"P{random.randint(9000, 9999):05d}" if r > .005 else None)  # orphan / null patient
        admit_s = None if random.random() < .005 else _fmt_dt(a["admit"])
        r = random.random()
        if r < .015:
            disch_s = _fmt_dt(a["admit"] - timedelta(days=random.randint(1, 3)))  # discharge BEFORE admission
        elif r < .035:
            disch_s = None
        else:
            disch_s = _fmt_dt(a["discharge"])
        r = random.random()
        sev_s = None if r < .04 else SEV_WORDS[a["severity"]] if r < .16 else str(a["severity"])
        rows.append((a["admission_id"], pid, _messy_dept(a["department"]),
                     None if random.random() < .03 else a["doctor_id"],
                     _messy_type(a["admission_type"]), None if random.random() < .02 else a["primary_diagnosis"], sev_s,
                     admit_s, disch_s, _fmt_bill(a["bill"]), _messy_outcome(a["outcome"])))
    rows += random.sample(rows, int(0.025 * len(rows)))  # double-submitted records
    random.shuffle(rows)

    con = sqlite3.connect(C.RAW / "hospital_his.db")
    con.execute("DROP TABLE IF EXISTS admissions")
    con.execute("""CREATE TABLE admissions (admission_id TEXT, patient_id TEXT, department TEXT, doctor_id TEXT,
                   admission_type TEXT, primary_diagnosis TEXT, severity TEXT, admit_date TEXT, discharge_date TEXT,
                   total_bill TEXT, outcome TEXT)""")
    con.executemany("INSERT INTO admissions VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    n_rows = con.execute("SELECT COUNT(*) FROM admissions").fetchone()[0]
    con.close()
    return adm_true, n_rows


# ───────────────────────── vitals (REST API) ─────────────────────────
def _ri(v):
    return int(round(float(v)))


def _rf(v, n=1):
    return round(float(v), n)


def _vitals_truth(truth, adm, k, n):
    s = (adm["severity"] - 1) / 4 * (1 - 0.6 * k / max(n - 1, 1))  # sicker -> worse vitals, improving over the stay
    diabetic = truth[adm["patient_id"]]["dia"]
    return dict(hr=np.random.normal(74 + 28 * s, 7), sys=np.random.normal(120 + 22 * s, 10),
                dia=np.random.normal(79 + 8 * s, 6), temp=np.random.normal(36.8 + 1.3 * s, 0.25),
                spo2=np.clip(np.random.normal(97.5 - 7 * s, 1.2), 70, 100),
                glu=np.random.normal((150 if diabetic else 100) + 35 * s, 18), resp=np.random.normal(16 + 6 * s, 2))


def _gen_vitals(truth, adm_true):
    readings = []
    for adm in adm_true:
        n = int(np.clip(adm["los"] + 1, 2, 6))
        span = (adm["discharge"] - adm["admit"]) / n
        for k in range(n):
            t = _vitals_truth(truth, adm, k, n)
            ts = adm["admit"] + span * k + timedelta(minutes=random.randint(0, 90))
            r = random.random()
            recorded = None if r < .01 else int(ts.replace(tzinfo=IST).timestamp()) if r < .07 else ts.strftime("%Y-%m-%dT%H:%M:%S")

            r = random.random()
            hr = None if r < .03 else random.choice([0, 300]) if r < .045 else f"{_ri(t['hr'])} bpm" if r < .095 else _ri(t["hr"])
            sys_, dia_ = _ri(t["sys"]), _ri(t["dia"])
            if random.random() < .01:
                sys_, dia_ = dia_, sys_  # swapped BP fields
            sys_ = None if random.random() < .03 else sys_
            dia_ = None if random.random() < .03 else dia_
            unit = "F" if random.random() < .15 else "C"
            temp = _rf(t["temp"] * 9 / 5 + 32) if unit == "F" else _rf(t["temp"])
            r = random.random()
            temp = None if r < .03 else (_rf(temp * 10) if r < .04 else temp)  # decimal-slip error
            r = random.random()
            spo2 = None if r < .03 else random.choice([0, 101, 127]) if r < .045 else f"{_ri(t['spo2'])}%" if r < .095 else _ri(t["spo2"])
            r = random.random()
            glu = None if r < .06 else "N/A" if r < .08 else _ri(t["glu"])
            resp = None if random.random() < .04 else _ri(t["resp"])
            r = random.random()
            pid = None if r < .01 else adm["patient_id"]
            aid = "A999999" if random.random() < .01 else adm["admission_id"]  # reading for an unknown admission
            dev = random.choice(DEVICES)
            dev = dev.upper() if random.random() < .05 else dev
            readings.append(dict(patient_id=pid, admission_id=aid, recorded_at=recorded, heart_rate=hr, systolic_bp=sys_,
                                 diastolic_bp=dia_, temperature=temp, temp_unit=unit, spo2=spo2, blood_glucose=glu,
                                 respiration_rate=resp, source_device=dev))

    readings += random.sample(readings, int(0.03 * len(readings)))  # API replays some records (same objects -> true duplicates)
    random.shuffle(readings)
    for i, rd in enumerate(readings, 1):
        rd["reading_id"] = f"R{i:07d}"

    total_pages = -(-len(readings) // C.PAGE_SIZE)

    def mock_vitals_api(page):
        """Stand-in for GET /api/v1/vitals?page=N - same JSON envelope a real API would return."""
        chunk = readings[(page - 1) * C.PAGE_SIZE: page * C.PAGE_SIZE]
        return {"status": "ok", "page": page, "total_pages": total_pages, "data": chunk}

    # the ingestion client: walk through every page and land the raw responses on disk
    page = 1
    while True:
        resp = mock_vitals_api(page)
        (C.RAW / "vitals_api" / f"vitals_page_{page:03d}.json").write_text(json.dumps(resp))
        if page >= resp["total_pages"]:
            break
        page += 1
    return len(readings), total_pages


def run() -> dict:
    """Generate all three raw sources and record how many rows each one delivered."""
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    truth, n_patients = _gen_patients()
    adm_true, n_adm = _gen_admissions(truth)
    n_vitals, pages = _gen_vitals(truth, adm_true)
    counts = {"patients": n_patients, "admissions": n_adm, "vitals": n_vitals, "vitals_pages": pages}
    (C.RAW / "_source_counts.json").write_text(json.dumps(counts))
    print(f"[generate] patients_registry.csv={n_patients:,}  hospital_his.db/admissions={n_adm:,}  "
          f"vitals API={n_vitals:,} rows in {pages} pages")
    return counts
