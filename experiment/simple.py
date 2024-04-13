"""Simple experiment over 3D simplex."""
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from util import (
    NSimplex,
    NSphere,
    OTSampler,
    ProductMLP,
    ot_train_step,
    save_plot,
    set_seeds,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_dirichlet_3d(points, file: str):
    v_a = torch.Tensor([[0, 1.0]])
    v_b = torch.Tensor([[-0.5, 0]])
    v_c = torch.Tensor([[0.5, 0]])
    points = points[:, 0].unsqueeze(-1) * v_a + points[:, 1].unsqueeze(-1) * v_b + points[:, 2].unsqueeze(-1) * v_c
    plt.scatter(points[:, 0], points[:, 1])
    save_plot(f"./out/{file}.png")


def plot(points, points_b):
    v_a = torch.Tensor([[0, 1.0]])
    v_b = torch.Tensor([[-0.5, 0]])
    v_c = torch.Tensor([[0.5, 0]])
    points = points[:, 0].unsqueeze(-1) * v_a + points[:, 1].unsqueeze(-1) * v_b + points[:, 2].unsqueeze(-1) * v_c
    plt.scatter(points[:, 0], points[:, 1])

    v_a = torch.Tensor([[0, 1.0]])
    v_b = torch.Tensor([[-0.5, 0]])
    v_c = torch.Tensor([[0.5, 0]])
    points_b = points_b[:, 0].unsqueeze(-1) * v_a + points_b[:, 1].unsqueeze(-1) * v_b + points_b[:, 2].unsqueeze(-1) * v_c
    plt.scatter(points_b[:, 0], points_b[:, 1])
    save_plot("./out/compare.png")


def generate_smiley_face(points: int):
    # Each eye is a concentrated dirichlet that is translated;
    # the smile is an arc
    v = np.array([[0, 0], [1, 0], [0.5, 1]])
    def area(vertices):
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]
        return 0.5 * np.abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))

    def barycentric_coordinates(point, vertices):
        area_ABC = area(vertices)
        alpha = area(np.array([point, vertices[1], vertices[2]])) / area_ABC
        beta = area(np.array([point, vertices[2], vertices[0]])) / area_ABC
        gamma = area(np.array([point, vertices[0], vertices[1]])) / area_ABC
        return alpha, beta, gamma

    def generate_eye(left: bool):
        if left:
            return np.random.dirichlet([50, 50, 30])
        return np.random.dirichlet([50, 10, 50])

    def generate_arc(centre, rad=1/4, ang=np.pi/6):
        r = np.random.rand() * 2 * ang + (ang / 2.0)
        return np.array([rad * np.cos(r), rad * np.sin(r)]) + centre

    import random
    pts = []
    for _ in range(points):
        i = random.randint(0, 2)
        if i <= 1:
            pts += [torch.Tensor(generate_eye(i == 0))]
        else:
            pt = generate_arc(np.array([0.35, 0.45]))
            pt = barycentric_coordinates(pt, v)
            pts += [torch.Tensor(pt)]
    return torch.stack(pts).view((points, 1, 3))


def run_simple_experiment(args: dict[str, Any]):
    set_seeds(1523)
    torch.autograd.set_detect_anomaly(True)
    model = ProductMLP(3, 1, 128, 4, activation="lrelu")
    model = model.to(device)
    epochs = 50
    manifold = NSimplex() if args["manifold"] == "simplex" else NSphere()
    sampler = OTSampler(manifold, "exact")
    raw = generate_smiley_face(10000)
    plot_dirichlet_3d(raw.squeeze(), "original")
    if args["manifold"] == "sphere":
        raw = NSimplex().sphere_map(raw)
    dataset = torch.utils.data.TensorDataset(raw)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(epochs):
        losses = []
        for x in train_loader:
            x = x[0].to(device)
            loss = ot_train_step(x, manifold, model, sampler)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += [loss.item()]
        print(f"Epoch {i+1} --- Loss {np.mean(losses)}")
    with torch.no_grad():
        x_0 = manifold.uniform_prior(1000, 1, 3).to(device)
        output = manifold.tangent_euler(x_0, model, 100)
    if args["manifold"] == "sphere":
        output = NSimplex().inv_sphere_map(output)
    print(output.min(), output.max())
    plot_dirichlet_3d(output.squeeze().cpu(), "generated")
