import argparse
import pandas as pd

REQUIRED = [
    "YEAR","MONTH","DAY_OF_MONTH","DAY_OF_WEEK",
    "ORIGIN","DEST",
    "CRS_DEP_TIME","DEP_TIME","DEP_DELAY",
    "CRS_ARR_TIME","ARR_TIME","ARR_DELAY"
]
def main(input_csv, output_csv):
    df = pd.read_csv(input_csv, low_memory=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for c in ["ORIGIN","DEST"]:
        for c in ["ORIGIN", "DEST"]:
            df[c] = df[c].astype(str).str.strip().str.upper()
    df = df.rename(columns={
        "DAY_OF_MONTH":"DAY",
        "ORIGIN":"ORG_AIRPORT",
        "DEST":"DEST_AIRPORT",
        "CRS_DEP_TIME":"SCHEDULED_DEPARTURE",
        "DEP_TIME":"DEPARTURE_TIME",
        "DEP_DELAY":"DEPARTURE_DELAY",
        "CRS_ARR_TIME":"SCHEDULED_ARRIVAL",
        "ARR_TIME":"ARRIVAL_TIME",
        "ARR_DELAY":"ARRIVAL_DELAY"
    })
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()
    main(args.input_csv, args.output_csv)
