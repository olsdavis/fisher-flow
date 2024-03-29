"""We define here an experiment analogous to that found in the DFM paper."""
import torch
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from util import (
    MLP,
    NSimplex,
    OTSampler,
    dfm_train_step,
    estimate_categorical_kl,
    generate_dirichlet_product,
    ot_train_step,
    set_seeds,
)


def _generate_dataset(probas: Tensor, n_train: int, n_test: int) -> tuple[Dataset, Dataset]:
    dist = Categorical(probas)
    x_train = torch.nn.functional.one_hot(dist.sample_n(n_train), probas.size(-1))
    x_test = torch.nn.functional.one_hot(dist.sample_n(n_test), probas.size(-1))
    return TensorDataset(x_train), TensorDataset(x_test)


def train(
    epochs: int,
    lr: float,
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    train_method: str,
    wasserstein_every: int = 10,
):
    print(f"===== {model_name} / {train_method}")
    set_seeds()
    manifold = NSimplex()
    ot_sampler = OTSampler(manifold, "exact")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    per_epoch_train = []
    per_epoch_test = []
    w2s = []
    for epoch in range(epochs):
        # Wasserstein eval
        if epoch % wasserstein_every == 0:
            model.eval()
            with torch.no_grad():
                shape = test_loader.dataset.tensors[0].shape
                n = shape[0]
                k = shape[1]
                d = shape[2]
                final_traj = manifold.tangent_euler(generate_dirichlet_product(n, k, d), model, 100)
                test = torch.cat(test_loader.dataset.tensors)
                w2 = manifold.wasserstein_dist(test, final_traj, power=2)
                w2s.append(w2)
            print(f"--- Epoch {epoch:03d}/{epochs+1:03d}: W2 distance = {w2:.5f}")

        # Training
        model.train()
        train_loss = []
        for x in train_loader:
            x = x[0]
            optimizer.zero_grad()
            if train_method == "ot-cft":
                loss = ot_train_step(x, manifold, model, ot_sampler)
            else:
                # dfm method otherwise
                loss = dfm_train_step(x, model)
            loss.backward()
            optimizer.step()
            train_loss += [loss.item()]
        per_epoch_train += [np.mean(train_loss)]

        # test loss
        test_loss = []
        model.eval()
        with torch.no_grad():
            for x in test_loader:
                x = x[0]
                optimizer.zero_grad()
                if train_method == "ot-cft":
                    loss = ot_train_step(x, manifold, model, ot_sampler)
                else:
                    # dfm method otherwise
                    loss = dfm_train_step(x, model)
                test_loss += [loss.item()]
        per_epoch_test += [np.mean(test_loss)]
        print(f"--- Epoch {epoch+1:03d}/{epochs:03d}: train loss = {per_epoch_train[-1]};"\
              f"test loss = {per_epoch_test[-1]}")
    plt.plot(np.arange(1, epochs+1), per_epoch_train, label="Train")
    plt.plot(np.arange(1, epochs+1), per_epoch_test, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"./out/dfm_toy_losses_{model_name}.pdf", bbox_inches="tight")
    plt.plot(np.arange(1, len(w2s) + 1) * wasserstein_every, w2s)
    plt.xlabel("Epoch")
    plt.ylabel("W2 distance")
    plt.savefig(f"./out/dfm_toy_w2_{model_name}.pdf", bbox_inches="tight")


def run_dfm_toy_experiment():
    set_seeds()
    # 3 3-simplex problem
    real_probas = torch.Tensor([[0.1, 0.3, 0.6], [0.4, 0.4, 0.2], [0.3, 0.4, 0.3]])
    train_dataset, test_dataset = _generate_dataset(real_probas, 10000, 1000)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    train(
        50, 1e-3, MLP(3, 4, 64, activation="lrelu"),
        "MLP OT-CFT", train_loader, test_loader, "ot-cft",
    )
    # print(estimate_categorical_kl())


if __name__ == "__main__":
    run_dfm_toy_experiment()
