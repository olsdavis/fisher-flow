"""Entry point."""
import argparse
from experiment import run_dfm_toy_experiment


def main():
    parser = argparse.ArgumentParser(
        prog="sfm_experiments",
        description="Experiments for SFM paper",
    )
    parser.add_argument("--experiment", "-e")
    args = parser.parse_args()

    experiments_available = {
        "dfm_toy": run_dfm_toy_experiment,
    }
    experiment = experiments_available["dfm_toy"]
    # if experiment found
    experiment(args)


if __name__ == "__main__":
    main()
