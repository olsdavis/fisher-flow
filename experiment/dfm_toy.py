"""We define here an experiment analogous to that found in the DFM paper."""
from typing import Any
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import wandb
import ipdb
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


class ToyDataset(torch.utils.data.IterableDataset):
    """
    Adapted from `https://github.com/HannesStark/dirichlet-flow-matching/blob/main/utils/dataset.py`.
    """
    def __init__(self, manifold, num_cls: int, toy_seq_len: int, toy_simplex_dim: int, sz: int = 100_000):
        super().__init__()
        self.m = manifold
        self.num_cls = num_cls
        self.sz = sz
        self.seq_len = toy_seq_len
        self.alphabet_size = toy_simplex_dim
        self.probs = torch.softmax(torch.rand((self.num_cls, self.seq_len, self.alphabet_size)), dim=2)
        self.class_probs = torch.ones(self.num_cls)
        if self.num_cls > 1:
            self.class_probs = self.class_probs * 1 / 2 / (self.num_cls - 1)
            self.class_probs[0] = 1 / 2
        assert self.class_probs.sum() == 1

    def __len__(self):
        return self.sz

    def __iter__(self):
        while True:
            cls = np.random.choice(a=self.num_cls,size=1,p=self.class_probs)
            seq = []
            for i in range(self.seq_len):
                sample = torch.multinomial(replacement=True,num_samples=1,input=self.probs[cls,i,:])
                one_hot = nn.functional.one_hot(sample, self.alphabet_size).float()
                seq.append(self.m.smooth_labels(one_hot, 0.81 if isinstance(self.m, NSphere) else 0.9))
                # seq.append(torch.multinomial(replacement=True,num_samples=1,input=self.probs[cls,i,:]))
            # yield torch.tensor(seq), cls
            yield torch.hstack(seq).squeeze(), cls


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _generate_raw_tensor(probas: Tensor, m: Manifold, n: int) -> Tensor:
    dist = Categorical(probas)
    x = nn.functional.one_hot(dist.sample((n,)), probas.size(-1))
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
    """
    Parameters:
        - `epochs`: the number of epochs to train for;
        - `lr`: the learning rate;
        - `model`: the model to train;
        - `train_loader`: the training data loader;
        - `test_loader`: the test data loader;
        - `wasserstein_set`: the set of points to evaluate the Wasserstein distance on;
        - `train_method`: the training method to use;
        - `wasserstein_every`: the number of epochs between Wasserstein evaluations;
        - `inference_steps`: the number of steps to take for inference;
        - `args`: the arguments.
    """
    print(f"===== {model} / {train_method}")
    set_seeds()
    model = model.to(device)
    manifold = NSimplex() if args["manifold"] == "simplex" else NSphere()
    ot_sampler = OTSampler(manifold, "exact")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = ReduceLROnPlateau(optimizer)
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    for epoch in range(epochs):
        logs = {}
        # Wasserstein eval
        if epoch % wasserstein_every == 0 and wasserstein_set:
            model.eval()
            with torch.inference_mode():
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
            # clip grad norm
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += [loss.item()]
        logs["train_loss"] = np.mean(train_loss)
        # lr_scheduler.step()
        lr_scheduler.step(logs["train_loss"])

        # test loss
        test_loss = []
        model.eval()
        with torch.inference_mode():
            i = 0
            for x in test_loader:
                if i > 1000:
                    break
                x = x[0]
                x = x.to(device)
                optimizer.zero_grad()
                if train_method == "ot-cft":
                    loss = ot_train_step(x, manifold, model, ot_sampler)
                else:
                    # dfm method otherwise
                    loss = dfm_train_step(x, model)
                test_loss += [loss.item()]
                i = i + 1
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
    n_train = 100_000
    model_config = load_model_config(args["config"])
    for d in ds:
        if args["wandb"]:
            wandb.init(
                project="simplex-flow-matching",
                name=f"toy_dfm_{d}",
                config={
                    "architecture": "ProductMLP",
                    "dataset": f"toy_dfm_{n_train}",
                },
            )
        set_seeds()

        manifold = NSimplex() if args["manifold"] == "simplex" else NSphere()

        # generate data
        # send to device for KL later
        real_probas = torch.softmax(torch.rand((seq_len, d)), dim=-1)
        train_dataset, test_dataset = _generate_dataset(real_probas, manifold, n_train, 10_000)
        # train_dataset = ToyDataset(manifold, num_cls=1, toy_seq_len=seq_len,
                # toy_simplex_dim=d, sz=100000)
        # test_dataset = train_dataset
        # wasserstein_set = _generate_raw_tensor(real_probas, manifold, 2500).to(device)
        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], num_workers=6, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], num_workers=6, shuffle=False)
        # train_loader = DataLoader(train_dataset, batch_size=512)
        # test_loader = DataLoader(test_dataset, batch_size=512)

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
