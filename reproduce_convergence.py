from utils.io import load_dataframe_folder
from utils.plot import show_training_progress

import argparse


def main(args):
    df = load_dataframe_folder(args.data_dir)
    show_training_progress(df, hue='model', metric=args.metric, name=args.save_path, save=True)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Plot Convergence")

    parser.add_argument('--data_dir', dest='data_dir', default="tables/CD_convergence")
    parser.add_argument('--metric', dest='metric', default="NDCG")
    parser.add_argument('--save_path', dest='save_path', default="figurename")

    args = parser.parse_args()

    main(args)
