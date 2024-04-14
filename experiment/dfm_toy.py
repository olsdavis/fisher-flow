"""We define here an experiment analogous to that found in the DFM paper."""
from typing import Any
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import wandb
from config import (
    load_model_config,
    model_from_config,
)
from util import (
    Manifold,
    NSimplex,
    NSphere,
    OTSampler,
    dfm_train_step,
    estimate_categorical_kl,
    ot_train_step,
    reset_memory,
    set_seeds,
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _generate_raw_tensor(probas: Tensor, m: Manifold, n: int) -> Tensor:
    dist = Categorical(probas)
    x = torch.nn.functional.one_hot(dist.sample((n,)), probas.size(-1))
    x = m.smooth_labels(x.float(), 0.81 if isinstance(m, NSphere) else 0.9)
    return x


def _generate_dataset(probas: Tensor, manifold: Manifold, n_train: int, n_test: int) -> tuple[Dataset, Dataset]:
    x_train = _generate_raw_tensor(probas, manifold, n_train)
    x_test = _generate_raw_tensor(probas, manifold, n_test)
    return TensorDataset(x_train), TensorDataset(x_test)


def train(
    epochs: int,
    lr: float,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    wasserstein_set: Tensor | None,
    train_method: str,
    wasserstein_every: int = 10,
    inference_steps: int = 100,
    args: dict[str, Any] = {},
) -> nn.Module:
    print(f"===== {model} / {train_method}")
    set_seeds()
    model = model.to(device)
    manifold = NSimplex() if args["manifold"] == "simplex" else NSphere()
    ot_sampler = OTSampler(manifold, "exact")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    lr_scheduler = ReduceLROnPlateau(optimizer)
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    for epoch in range(epochs):
        logs = {}
        # Wasserstein eval
        if epoch % wasserstein_every == 0 and wasserstein_set:
            model.eval()
            with torch.no_grad():
                shape = wasserstein_set.shape
                n = shape[0]
                k = shape[1]
                d = shape[2]
                x_0 = manifold.uniform_prior(n, k, d).to(device)
                final_traj = manifold.tangent_euler(x_0, model, inference_steps)
                w2 = manifold.wasserstein_dist(wasserstein_set, final_traj, power=2)
            logs["w2"] = w2
            print(f"--- Epoch {epoch+1:03d}/{epochs:03d}: W2 distance = {w2:.5f}")

        # Training
        model.train()
        train_loss = []
        for x in train_loader:
            x = x[0]
            x = x.to(device)
            # x is one-hot encoded so fixed-point of either sphere-map or inverse
            optimizer.zero_grad()
            if train_method == "ot-cft":
                loss = ot_train_step(x, manifold, model, ot_sampler)
            else:
                # dfm method otherwise
                loss = dfm_train_step(x, model)
            loss.backward()
            optimizer.step()
            train_loss += [loss.item()]
        logs["train_loss"] = np.mean(train_loss)
        # lr_scheduler.step()
        lr_scheduler.step(logs["train_loss"])

        # test loss
        test_loss = []
        model.eval()
        with torch.no_grad():
            for x in test_loader:
                x = x[0]
                x = x.to(device)
                optimizer.zero_grad()
                if train_method == "ot-cft":
                    loss = ot_train_step(x, manifold, model, ot_sampler)
                else:
                    # dfm method otherwise
                    loss = dfm_train_step(x, model)
                test_loss += [loss.item()]
        logs["test_loss"] = np.mean(test_loss)
        print(f"--- Epoch {epoch+1:03d}/{epochs:03d}: train loss = {np.mean(train_loss):.5f};"\
              f" test loss = {np.mean(test_loss):.5f}")
        if args["wandb"]:
            wandb.log(logs)
    return model


def run_dfm_toy_experiment(args: dict[str, Any]):
    """
    Runs the experiment on the toy data.
    """
    kls = []
    seq_len = 4
    epochs = 1000
    lr = 1e-3
    ds = [5, 10, 20, 40, 60, 80, 100, 120, 140, 160]
    model_config = load_model_config(args["config"])
    for d in ds:
        if args["wandb"]:
            wandb.init(
                project="simplex-flow-matching",
                name=f"toy_dfm_{d}",
                config={
                    "architecture": "ProductMLP",
                    "dataset": "toy_dfm",
                },
            )
        set_seeds()

        manifold = NSimplex() if args["manifold"] == "simplex" else NSphere()
        print(type(manifold).__name__)

        # generate data
        # send to device for KL later
        real_probas = torch.softmax(torch.rand((seq_len, d)), dim=-1)
        train_dataset, test_dataset = _generate_dataset(real_probas, manifold, 10000, 1000)
        # wasserstein_set = _generate_raw_tensor(real_probas, manifold, 2500).to(device)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        # start
        print(f"---=== {d}-simplices ===---")
        trained = train(
            epochs,
            lr,
            model_from_config(
                k=seq_len,
                dim=d,
                config=model_config,
            ),
            train_loader,
            test_loader,
            wasserstein_set=None,
            train_method=args["train_method"],
            inference_steps=args["inference_steps"],
            args=args,
        )

        # evaluate KL
        kls += [
            estimate_categorical_kl(
                trained,
                manifold,
                real_probas.to(device),
                n=args["kl_points"],
                inference_steps=args["inference_steps"],
                sampling_mode=args["sampling_mode"],
            )
        ]

        del train_dataset
        del test_dataset
        reset_memory()
        print(f"KL {kls[-1]:.5f}")

        # log that single value
        if args["wandb"]:
            wandb.log({"kl": kls[-1]})
            wandb.finish()
