"""Entry point."""
import argparse
from experiment import run_dfm_toy_experiment


def main():
    # args
    parser = argparse.ArgumentParser(
        prog="sfm_experiments",
        description="Experiments for SFM paper",
    )
    parser.add_argument("--experiment", "-e")
    parser.add_argument("--inference_steps", default=100, type=int)
    parser.add_argument("--kl_points", default=10000, type=int)
    parser.add_argument("--sampling_mode", default="max", type=str)
    parser.add_argument("--train_method", default="ot-cft", type=str)
    args = vars(parser.parse_args())
    print(args)

    experiments_available = {
        "dfm_toy": run_dfm_toy_experiment,
    }
    experiment = experiments_available["dfm_toy"]
    # if experiment found
    experiment(args)


if __name__ == "__main__":
    main()
