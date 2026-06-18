import random
import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INPUT_FILE  = "nyc_taxi_data_1000.csv"
OUTPUT_FILE = "nyc_taxi_data_1000_invalid.csv"
REPORT_FILE = "alteration_report.json"

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
print(f"Loaded {len(df)} rows from {INPUT_FILE}")
df_invalid   = df.copy()
alterations  = []
used_rows    = set()
def pick(n, exclude):
    """Return n row indices not already in `exclude`."""
    pool = list(set(range(len(df))) - exclude)
    chosen = random.sample(pool, n)
    exclude.update(chosen)
    return chosen
def record(idx, field, original, altered, violation):
    alterations.append({
        "row":       int(idx),
        "field":     field,
        "original":  str(original),
        "altered":   str(altered),
        "violation": violation,
    })
# ── 1. VendorID out of range ─────────────────────────────────────────────────
rows = pick(20, used_rows)
for r in rows[:10]:
    df_invalid.at[r, "VendorID"] = 3
    record(r, "VendorID", df.at[r, "VendorID"], 3,
           "Value 3 is outside the valid range [1, 2]")
for r in rows[10:]:
    df_invalid.at[r, "VendorID"] = 0
    record(r, "VendorID", df.at[r, "VendorID"], 0,
           "Value 0 is outside the valid range [1, 2]")
print(f"  Cat 1 – VendorID:              20 rows altered")
# ── 2. passenger_count out of range ─────────────────────────────────────────
rows = pick(15, used_rows)
for r in rows[:8]:
    df_invalid.at[r, "passenger_count"] = -1
    record(r, "passenger_count", df.at[r, "passenger_count"], -1,
           "Value −1 is below the valid minimum of 0")
for r in rows[8:]:
    df_invalid.at[r, "passenger_count"] = 10
    record(r, "passenger_count", df.at[r, "passenger_count"], 10,
           "Value 10 exceeds the valid maximum of 9")
print(f"  Cat 2 – passenger_count:       15 rows altered")
# ── 3. trip_distance negative ────────────────────────────────────────────────
rows = pick(10, used_rows)
for r in rows:
    orig = float(df.at[r, "trip_distance"])
    val  = -abs(orig) if orig != 0 else -0.1
    df_invalid.at[r, "trip_distance"] = round(val, 2)
    record(r, "trip_distance", orig, round(val, 2),
           "Negative trip distance violates valid minimum of 0")
print(f"  Cat 3 – trip_distance:         10 rows altered")
# ── 4. RateCodeID out of range ───────────────────────────────────────────────
rows = pick(10, used_rows)
for r in rows[:5]:
    df_invalid.at[r, "RateCodeID"] = 7
    record(r, "RateCodeID", df.at[r, "RateCodeID"], 7,
           "Value 7 is outside the valid range [1, 6]")
for r in rows[5:]:
    df_invalid.at[r, "RateCodeID"] = 0
    record(r, "RateCodeID", df.at[r, "RateCodeID"], 0,
           "Value 0 is outside the valid range [1, 6]")
print(f"  Cat 4 – RateCodeID:            10 rows altered")
# ── 5. store_and_fwd_flag invalid enum ──────────────────────────────────────
rows = pick(10, used_rows)
for r in rows[:5]:
    df_invalid.at[r, "store_and_fwd_flag"] = "X"
    record(r, "store_and_fwd_flag", df.at[r, "store_and_fwd_flag"], "X",
           "Value 'X' not in valid set {Y, N}")
for r in rows[5:]:
    df_invalid.at[r, "store_and_fwd_flag"] = "YES"
    record(r, "store_and_fwd_flag", df.at[r, "store_and_fwd_flag"], "YES",
           "Value 'YES' not in valid set {Y, N}")
print(f"  Cat 5 – store_and_fwd_flag:    10 rows altered")

# ── 6. payment_type out of range ─────────────────────────────────────────────
rows = pick(8, used_rows)
for r in rows[:4]:
    df_invalid.at[r, "payment_type"] = 7
    record(r, "payment_type", df.at[r, "payment_type"], 7,
           "Value 7 is outside the valid range [1, 6]")
for r in rows[4:]:
    df_invalid.at[r, "payment_type"] = 0
    record(r, "payment_type", df.at[r, "payment_type"], 0,
           "Value 0 is outside the valid range [1, 6]")
print(f"  Cat 6 – payment_type:          8 rows altered")
# ── 7. fare_amount negative ──────────────────────────────────────────────────
rows = pick(10, used_rows)
for r in rows:
    orig = float(df.at[r, "fare_amount"])
    val  = -abs(orig) if orig != 0 else -1.0
    df_invalid.at[r, "fare_amount"] = round(val, 2)
    record(r, "fare_amount", orig, round(val, 2),
           "Negative fare_amount violates valid minimum of 0")
print(f"  Cat 7 – fare_amount:           10 rows altered")
# ── 8. mta_tax invalid discrete value ───────────────────────────────────────
rows = pick(8, used_rows)
for r in rows:
    df_invalid.at[r, "mta_tax"] = 0.25
    record(r, "mta_tax", df.at[r, "mta_tax"], 0.25,
           "Value 0.25 not in valid discrete set {0.0, 0.5}")
print(f"  Cat 8 – mta_tax:               8 rows altered")
# ── 9. improvement_surcharge invalid discrete value ──────────────────────────
rows = pick(8, used_rows)
for r in rows:
    df_invalid.at[r, "improvement_surcharge"] = 0.50
    record(r, "improvement_surcharge", df.at[r, "improvement_surcharge"], 0.50,
           "Value 0.50 not in valid discrete set {0.0, 0.3}")
print(f"  Cat 9 – improvement_surcharge: 8 rows altered")
# ── 10. Missing values in required fields ────────────────────────────────────
NULL_FIELDS = ["fare_amount", "total_amount", "passenger_count",
               "trip_distance", "VendorID"]
rows = pick(10, used_rows)
for i, r in enumerate(rows):
    field = NULL_FIELDS[i % len(NULL_FIELDS)]
    orig  = df.at[r, field]
    df_invalid.at[r, field] = np.nan
    record(r, field, orig, None,
           f"Required field '{field}' set to NULL (missing value)")
print(f"  Cat 10 – Null required fields: 10 rows altered")
# ── 11. total_amount negative ────────────────────────────────────────────────
rows = pick(7, used_rows)
for r in rows:
    orig = float(df.at[r, "total_amount"])
    val  = -abs(orig) if orig != 0 else -1.0
    df_invalid.at[r, "total_amount"] = round(val, 2)
    record(r, "total_amount", orig, round(val, 2),
           "Negative total_amount violates valid minimum of 0")
print(f"  Cat 11 – total_amount:         7 rows altered")
# ── Save corrupted dataset ───────────────────────────────────────────────────
df_invalid.to_csv(OUTPUT_FILE, index=False)
print(f"\nCorrupted dataset saved: {OUTPUT_FILE}")
