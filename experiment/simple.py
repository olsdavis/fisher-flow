"""Simple experiment over 3D simplex."""
import random
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from util import (
    NSimplex,
    NSphere,
    OTSampler,
    ProductMLP,
    ot_train_step,
    save_plot,
    set_seeds,
)
from geoopt import ProductManifold, Euclidean, Sphere
import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _output_and_div(vecfield, x, v=None, div_mode="exact"):
    # From: https://github.com/facebookresearch/riemannian-fm/blob/main/manifm/model_pl.py#L45
    def div_fn(u):
        """Accepts a function u:R^D -> R^D."""
        J = torch.func.jacrev(u)
        return lambda x: torch.trace(J(x))
    if div_mode == "exact":
        dx = vecfield(x)
        div = torch.vmap(div_fn(vecfield))(x)
    else:
        dx, vjpfunc = torch.func.vjp(vecfield, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, div


def plot_dirichlet_3d(points, file: str):
    pts = simplex_to_cartesian(points.cpu().numpy())
    fig = plt.figure()
    plt.scatter(pts[:, 0], pts[:, 1])
    draw_triangle()
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


def __generate_smiley_face(points: int):
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



def ___generate_smiley_face(points: int):
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
            centre = np.array([0.410, 0.53])
        else:
            centre = np.array([0.45, 0.45])
        rad = np.random.uniform(0, 0.01)
        point = np.random.uniform(0, 2 * np.pi)
        return np.array([rad * np.cos(point), rad * np.sin(point)]) + centre

    def generate_arc(centre, rad=1/6, ang=np.pi/6, ang_rad=0.65):
        r = (np.random.rand() * 2 * ang + (ang / 2.0)) * ang_rad
        return np.array([rad * np.cos(r), rad * np.sin(r)]) + centre

    pts = []
    for _ in range(points):
        i = random.randint(0, 2)
        if i <= 1:
            pts += [
                torch.Tensor(barycentric_coordinates(torch.Tensor(generate_eye(i == 0)), v))
            ]
        else:
            pt = generate_arc(np.array([0.35, 0.45]))
            pt = barycentric_coordinates(pt, v)
            pts += [torch.Tensor(pt)]
    return torch.stack(pts).view((points, 1, 3))


def cartesian_to_3simplex(x, y):
    # Vertices of the 3-simplex (triangle) in 2D space
    V = np.array([
        [0, 0],    # Vertex A
        [1, 0],    # Vertex B
        [0.5, np.sqrt(3)/2]  # Vertex C
    ])
    
    # Matrix of vertex coordinates
    M = np.vstack((V.T, np.ones(3)))
    
    # Point in homogenous coordinates
    P = np.array([x, y, 1])
    
    # Solve linear system to find barycentric coordinates
    bary_coords = np.linalg.solve(M, P)
    
    return bary_coords


def simplex_to_cartesian(ps):
    ret = []
    for p in ps:
        V = np.array([
            [0, 0],    # Vertex A
            [1, 0],    # Vertex B
            [0.5, np.sqrt(3)/2]  # Vertex C
        ])
        ret += [p[0] * V[0, :] + p[1] * V[1, :] + p[2] * V[2, :]]
    return np.array(ret)


def generate_smiley_face(points: int):
    def circle_uniform(n_points: int, center: np.ndarray, rad: float = 0.05):
        rad = np.random.uniform(0, rad, n_points)[..., None]
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        points = rad * np.array([np.cos(theta), np.sin(theta)]).T
        return center + points

    def generate_uniform_arc(n_points: int, radius: float, center, angle: float):
        theta = np.linspace(0, angle, n_points)
        rad_pert = np.random.uniform(-0.03, 0.03, n_points)
        x = (radius + rad_pert) * np.cos(theta + 1.25 * np.pi)
        y = (radius + rad_pert) * np.sin(theta + 1.25 * np.pi)
        points = np.column_stack((x, y))
        return points + center[..., None].T
    pts = []
    split_a = points // 3
    split_b = points // 3
    split_c = points - split_a - split_b
    pts += [
        circle_uniform(split_a, np.array([0.4, 0.45]))
    ]
    pts += [
        circle_uniform(split_b, np.array([0.6, 0.45]))
    ]
    pts += [
        generate_uniform_arc(split_c, 0.2, np.array([0.5, 0.3]), np.pi/2)
    ]
    ret = []
    for xs in pts:
        for p in xs:
            ret += [torch.from_numpy(cartesian_to_3simplex(p[0], p[1]))]
    return torch.stack(ret).view((points, 1, 3)).float()


# The following is adapted from: https://github.com/facebookresearch/riemannian-fm
def _euler_step(odefunc, xt, vt, t0, dt, manifold=None):
    if manifold is not None:
        return manifold.expmap(xt, dt * vt)
    else:
        return xt + dt * vt


@torch.no_grad()
def _projx_integrator_return_last(
    manifold, odefunc, x0, t, method="euler", projx=True, local_coords=False, pbar=False
):
    """Has a lower memory cost since this doesn't store intermediate values."""

    step_fn = {
        "euler": _euler_step,
    }[method]

    xt = x0

    t0s = t[:-1]
    if pbar:
        t0s = tqdm.tqdm(t0s)

    for t0, t1 in zip(t0s, t[1:]):
        dt = t1 - t0
        vt = odefunc(t0, xt)
        xt = step_fn(
            odefunc, xt, vt, t0, dt, manifold=manifold if local_coords else None
        )
        if projx:
            xt = manifold.projx(xt)
    return xt


@torch.no_grad()
def compute_exact_loglikelihood(
    model: torch.nn.Module,
    batch: Tensor,
    manifold,
    t1: float = 1.0,
    num_steps: int = 1000,
    div_mode: str = "rademacher",
    local_coords: bool = False,
    eval_projx: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    normalize_loglikelihood: bool = False,
) -> Tensor:
    """Computes the negative log-likelihood of a batch of data."""
    # Based on https://github.com/facebookresearch/riemannian-fm/blob/main/manifm/model_pl.py#L449

    try:
        with torch.inference_mode(mode=False):
            v = None
            if div_mode == "rademacher":
                v = torch.randint(low=0, high=2, size=batch.shape).to(batch) * 2 - 1

            def odefunc(t, tensor):
                # t = t.to(tensor)
                x = tensor[..., : batch.size(-1)]
                vecfield = lambda x: model(x, t)
                dx, div = _output_and_div(vecfield, x, v=v, div_mode=div_mode)

                if hasattr(manifold, "logdetG"):

                    def _jvp(x, v):
                        return torch.func.jvp(manifold.logdetG, (x,), (v,))[1]

                    corr = torch.func.vmap(_jvp)(x, dx)
                    div = div + 0.5 * corr#.to(div)

                # div = div.view(-1, 1)
                div = div[..., None]
                del t, x
                return torch.cat([dx, div], dim=-1)

            # Solve ODE on the product manifold of data manifold x euclidean.
            product_man = ProductManifold(
                (manifold, batch.size(-1)), (Euclidean(), 1)
            )
            state1 = torch.cat([batch, torch.zeros_like(batch[..., :1])], dim=-1)

            with torch.no_grad():
                if not eval_projx and not local_coords:
                    # If no projection, use adaptive step solver.
                    state0 = odeint(
                        odefunc,
                        state1,
                        t=torch.linspace(0, t1, 2).to(batch),
                        atol=atol,
                        rtol=rtol,
                        method="dopri5",
                        options={"min_step": 1e-5},
                    )[-1]
                else:
                    # If projection, use 1000 steps.
                    state0 = _projx_integrator_return_last(
                        product_man,
                        odefunc,
                        state1,
                        t=torch.linspace(0, t1, num_steps + 1).to(batch),
                        method="euler",
                        projx=eval_projx,
                        local_coords=local_coords,
                        pbar=True,
                    )

            x0, logdetjac = state0[..., : batch.size(-1)], state0[..., -1]
            # x0_ = x0
            x0 = manifold.projx(x0).abs()

            # log how close the final solution is to the manifold.
            # integ_error = (x0[..., : self.dim] - x0_[..., : self.dim]).abs().max()
            # self.log("integ_error", integ_error)

            # logp0 = manifold.base_logprob(x0)
            logp0 = manifold.uniform_logprob(x0)
            logp1 = logp0 + logdetjac.sum(dim=-1)

            if normalize_loglikelihood:
                logp1 = logp1 / np.prod(batch.shape[1:])

            # Mask out those that left the manifold
            masked_logp1 = logp1
            return masked_logp1
    except:
        traceback.print_exc()
        return torch.zeros(batch.shape[0]).to(batch)




def draw_triangle():
    V = np.array([
        [0, 0],    # Vertex A
        [1, 0],    # Vertex B
        [0.5, np.sqrt(3)/2]  # Vertex C
    ])
    for i in range(3):
        nx = (i + 1) % 3
        plt.plot([V[i, 0], V[nx, 0]], [V[i, 1], V[nx, 1]], 'k-')


def generate_likelihood(points: int):
    raise NotImplementedError("")


def run_simple_experiment(args: dict[str, Any]):
    set_seeds(1523)
    torch.autograd.set_detect_anomaly(True)
    model = ProductMLP(3, 1, 128, 4, activation="lrelu")
    model = model.to(device)
    epochs = 50
    manifold = NSimplex() if args["manifold"] == "simplex" else NSphere()
    sampler = OTSampler(manifold, "exact") if args["ot"] else None
    raw = generate_smiley_face(10000)
    np.save(f"./out/original.npy", raw.cpu().numpy())
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
    for i, frame in enumerate(output):
        np.save(f"./out/generated_{type(manifold).__name__}_{type(sampler).__name__}_{i}.npy", frame.squeeze().cpu().numpy())
    plot_dirichlet_3d(output.squeeze().cpu(), "generated")
