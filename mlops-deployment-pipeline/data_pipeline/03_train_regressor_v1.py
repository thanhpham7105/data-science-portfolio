import argparse, os, sys

# allow CLI args for MLProject
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_alphas", type=int, default=20)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--cleaned_csv", type=str, default="data/cleaned_data.csv")
    # optional logs (not used in v1)
    parser.add_argument("--import_log", default="import_data.log")
    parser.add_argument("--clean_log", default="clean_data.log")
    return parser.parse_args()
args = parse_args()

num_alphas = args.num_alphas
order = args.order

if args.cleaned_csv != "cleaned_data.csv" and os.path.exists(args.cleaned_csv):
    import shutil
    shutil.copy(args.cleaned_csv, "cleaned_data.csv")

