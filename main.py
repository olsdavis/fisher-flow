"""Entry point."""
import argparse
from experiment import run_dfm_toy_experiment


def main():
    training_methods = ["ot-cft", "dft"]
    experiments_available = {
        "dfm_toy": run_dfm_toy_experiment,
    }

    # args
    parser = argparse.ArgumentParser(
        prog="sfm_experiments",
        description="Experiments for SFM paper",
    )
    # What experiment to run?
    parser.add_argument("--experiment", "-e", type=str, choices=experiments_available.keys())
    # How many steps to use in inference? (Used also for KL, for instance.)
    parser.add_argument("--inference_steps", default=100, type=int)
    # How many points to use in KL estimation?
    parser.add_argument("--kl_points", default=512_000, type=int)
    # How to sample points? Draw from distribution or take argmax proba?
    parser.add_argument("--sampling_mode", default="max", type=str)
    # What training method to use?
    parser.add_argument("--train_method", default="ot-cft", type=str, choices=training_methods)
    args = vars(parser.parse_args())
    print(args)

    experiment = experiments_available[args["experiment"]]
    # if experiment found
    experiment(args)


if __name__ == "__main__":
    main()
