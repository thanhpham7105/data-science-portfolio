import argparse, logging
import pandas as pd

def clip(x, lo=-120, hi=600):
    return x.clip(lower=lo, upper=hi)

def main(input_csv, airport, month, output_csv, log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode="w")
    df = pd.read_csv(input_csv)
    keep = [
        "YEAR","MONTH","DAY","DAY_OF_WEEK","ORG_AIRPORT","DEST_AIRPORT",
        "SCHEDULED_DEPARTURE","DEPARTURE_TIME","DEPARTURE_DELAY",
        "SCHEDULED_ARRIVAL","ARRIVAL_TIME","ARRIVAL_DELAY"
    ]
    df = df[keep]
    m = (df["ORG_AIRPORT"].astype(str).str.upper()==airport.upper()) & (df["MONTH"].astype(int)==int(month))
    out = df.loc[m].copy()

    out = out.dropna(subset=["DEPARTURE_DELAY"])
    if "DEPARTURE_DELAY" in out: out["DEPARTURE_DELAY"] = clip(out["DEPARTURE_DELAY"])
    if "ARRIVAL_DELAY" in out:   out["ARRIVAL_DELAY"] = clip(out["ARRIVAL_DELAY"])
    out = out.drop_duplicates()

    out.to_csv(output_csv, index=False)
    logging.info("Wrote cleaned CSV to %s", output_csv)
    logging.info("Rows: %d; Unique DEST_AIRPORT: %d", len(out), out["DEST_AIRPORT"].nunique())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--airport", required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--log_file", default="clean_data.log")
    args = ap.parse_args()
    main(args.input_csv, args.airport, args.month, args.output_csv, args.log_file)
