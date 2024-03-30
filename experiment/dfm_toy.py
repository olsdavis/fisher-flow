"""We define here an experiment analogous to that found in the DFM paper."""
import torch
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import (
    MLP,
    ProductMLP,
    TembMLP,
    NSimplex,
    OTSampler,
    dfm_train_step,
    estimate_categorical_kl,
    generate_dirichlet_product,
    ot_train_step,
    set_seeds,
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def label_smoothing(one_hot_labels, smoothing=0.98):
    """
    Applies label smoothing to a batch of one-hot encoded vectors.

    Parameters:
    - one_hot_labels: A tensor of shape (batch_size, num_classes) containing one-hot encoded labels.
    - smoothing: The value to assign to the target class in each label vector. Default is 0.95.

    Returns:
    - A tensor of the same shape as one_hot_labels with smoothed labels.
    """
    num_classes = one_hot_labels.size(-1)
    
    # Value to be added to each non-target class
    increase = torch.tensor((1.0 - smoothing) / (num_classes - 1))
    
    # Create a tensor with all elements set to the increase value
    smooth_labels = torch.full_like(one_hot_labels.float(), increase)
    
    # Set the target classes to the smoothing value
    smooth_labels[one_hot_labels == 1] = smoothing
    
    return smooth_labels


def _generate_dataset(probas: Tensor, n_train: int, n_test: int) -> tuple[Dataset, Dataset]:
    dist = Categorical(probas)
    x_train = torch.nn.functional.one_hot(dist.sample((n_train,)), probas.size(-1))
    x_train = label_smoothing(x_train)
    x_test = torch.nn.functional.one_hot(dist.sample((n_test,)), probas.size(-1))
    x_test = label_smoothing(x_test)
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
    plot_run: bool = False,
) -> nn.Module:
    print(f"===== {model_name} / {train_method}")
    set_seeds()
    model = model.to(device)
    manifold = NSimplex()
    ot_sampler = OTSampler(manifold, "exact")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
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
                final_traj = manifold.tangent_euler(generate_dirichlet_product(n, k, d).to(device), model, 100)
                test = torch.cat(test_loader.dataset.tensors).to(device)
                w2 = manifold.wasserstein_dist(test, final_traj, power=2)
                w2s.append(w2)
            print(f"--- Epoch {epoch+1:03d}/{epochs:03d}: W2 distance = {w2:.5f}")

        # Training
        model.train()
        train_loss = []
        for x in train_loader:
            x = x[0]
            x = x.to(device)
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
        lr_scheduler.step()

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
        per_epoch_test += [np.mean(test_loss)]
        print(f"--- Epoch {epoch+1:03d}/{epochs:03d}: train loss = {per_epoch_train[-1]:.5f};"\
              f" test loss = {per_epoch_test[-1]:.5f}")
    if plot_run:
        plt.plot(np.arange(1, epochs+1), per_epoch_train, label="Train")
        plt.plot(np.arange(1, epochs+1), per_epoch_test, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim(left=1)
        plt.legend()
        plt.savefig(f"./out/dfm_toy_losses_{model_name}.pdf", bbox_inches="tight")
        plt.figure()
        plt.plot(np.arange(1, len(w2s) + 1) * wasserstein_every, w2s)
        plt.xlim(left=10)
        plt.xlabel("Epoch")
        plt.ylabel("W2 distance")
        plt.savefig(f"./out/dfm_toy_w2_{model_name}.pdf", bbox_inches="tight")
    return model


def run_dfm_toy_experiment():
    seq_len = 4
    kls = []
    mx = 160
    epochs = 1000
    for d in range(20, mx+1, 20):
        set_seeds()
        real_probas = torch.softmax(torch.rand((seq_len, d)), dim=-1).to(device)
        train_dataset, test_dataset = _generate_dataset(real_probas, 10000, 1000)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        print(f"---=== {real_probas.size(-1)}-simplices ===---")
        trained = train(
            epochs, 1e-3,
            # MLP(3, 4, 128, activation="relu"),
            ProductMLP(d, seq_len, 128, 4, activation="gelu"),
            # TembMLP(d, seq_len, hidden_layers=4),
            "ProductMLP OT-CFT", train_loader, test_loader, "ot-cft",
        )
        # sample lots of points
        with torch.no_grad():
            prior = torch.distributions.dirichlet.Dirichlet(
                torch.Tensor(torch.ones_like(real_probas)),
            )
            n_points = 10000
            x_0 = prior.sample((n_points,)).to(device)
            simplex = NSimplex()
            x_1 = simplex.tangent_euler(x_0, trained, 100)
            kls += [estimate_categorical_kl(x_1, real_probas)]
        print(f"KL {kls[-1]:.5f}")
    plt.figure()
    plt.plot(np.arange(20, mx+1, 20), kls)
    plt.xlabel("Categories")
    plt.ylabel("KL divergence")
    plt.savefig("./out/kl.pdf", bbox_inches="tight")


if __name__ == "__main__":
    run_dfm_toy_experiment()
