import argparse, pandas as pd

def main(input_csv, airport, month, output_csv):
    df = pd.read_csv(input_csv)
    mask = (df["ORG_AIRPORT"].astype(str).str.upper()==airport.upper()) & (df["MONTH"].astype(int)==int(month))
    out = df.loc[mask].copy()
    out = out.dropna(subset=["DEPARTURE_DELAY"])
    out.to_csv(output_csv, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--airport", required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()
    main(args.input_csv, args.airport, args.month, args.output_csv)
