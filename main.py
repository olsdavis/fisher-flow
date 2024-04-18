"""Entry point."""
import argparse
from experiment import (
    run_dfm_toy_experiment,
    run_simple_experiment,
)


def main():
    training_methods = ["ot-cft", "dft"]
    experiments_available = {
        "dfm_toy": run_dfm_toy_experiment,
        "simple": run_simple_experiment,
    }

    # args
    parser = argparse.ArgumentParser(
        prog="sfm_experiments",
        description="Experiments for SFM paper",
    )
    # What manifold to use?
    parser.add_argument("--manifold", "-m", type=str, default="simplex", choices=["simplex", "sphere"])
    # What experiment to run?
    parser.add_argument("--experiment", "-e", type=str, choices=experiments_available.keys())
    # How many steps to use in inference? (Used also for KL, for instance.)
    parser.add_argument("--inference_steps", default=100, type=int)
    # How many steps to use in inference? (Used also for KL, for instance.)
    parser.add_argument("--batch_size", default=2048, type=int)
    # How many points to use in KL estimation?
    parser.add_argument("--kl_points", "-kl", default=512_000, type=int)
    # How to sample points? Draw from distribution or take argmax proba?
    parser.add_argument("--sampling_mode", default="max", type=str)
    # What training method to use?
    parser.add_argument("--train_method", default="ot-cft", type=str, choices=training_methods)
    # Where is the config for the model?
    parser.add_argument("--config", "-c", type=str)
    # Output to wandb?
    parser.add_argument("--wandb", action="store_true")
    args = vars(parser.parse_args())
    print(args)

    experiment = experiments_available[args["experiment"]]
    # if experiment found
    experiment(args)


if __name__ == "__main__":
    main()
