import argparse
from utils.io import load_dataframe_folder, load_yaml
from plots.rec_plots import show_training_progress


def main(args):
    df = load_dataframe_folder(args.path)
    show_training_progress(df, hue='model', metric=args.metric, name=args.name, save=True)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="CNCF")
    parser.add_argument('-d', dest='path', default="tables/CD_convergence")
    parser.add_argument('-m', dest='metric', default="NDCG")
    parser.add_argument('-n', dest='name', default="figurename")
    args = parser.parse_args()

    main(args)
