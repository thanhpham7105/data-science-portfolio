import argparse, logging
import pandas as pd
import numpy as np

MAP = {
    "DAY_OF_MONTH":"DAY",
    "ORIGIN":"ORG_AIRPORT",
    "DEST":"DEST_AIRPORT",
    "CRS_DEP_TIME":"SCHEDULED_DEPARTURE",
    "DEP_TIME":"DEPARTURE_TIME",
    "DEP_DELAY":"DEPARTURE_DELAY",
    "CRS_ARR_TIME":"SCHEDULED_ARRIVAL",
    "ARR_TIME":"ARRIVAL_TIME",
    "ARR_DELAY":"ARRIVAL_DELAY"
}
REQUIRED = [
    "YEAR","MONTH","DAY_OF_MONTH","DAY_OF_WEEK",
    "ORIGIN","DEST","CRS_DEP_TIME","DEP_TIME","DEP_DELAY",
    "CRS_ARR_TIME","ARR_TIME","ARR_DELAY"
]

def to_int_hhmm(s):
    if pd.isna(s): return np.nan
    s = int(s)
    if s == 2400: s = 0
    return s

def main(input_csv, output_csv, log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode="w")
    df = pd.read_csv(input_csv, low_memory=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for c in ["ORIGIN","DEST"]:
        df[c] = df[c].astype(str).str.strip().str.upper()

    df = df.rename(columns=MAP)

    for c in ["SCHEDULED_DEPARTURE","DEPARTURE_TIME","SCHEDULED_ARRIVAL","ARRIVAL_TIME"]:
        df[c] = df[c].apply(to_int_hhmm).astype("Int64")

    for c in ["DEPARTURE_DELAY","ARRIVAL_DELAY"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df.to_csv(output_csv, index=False)
    logging.info("Wrote formatted CSV to %s", output_csv)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--log_file", default="import_data.log")
    args = ap.parse_args()
    main(args.input_csv, args.output_csv, args.log_file)
