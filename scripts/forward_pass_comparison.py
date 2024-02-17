# %% This script compares the forward pass of the published torch model with our implemented jax model
import random
from functools import partial

import jax
import numpy as np
import torch
import torch.nn as nn
from jax import Array, vmap
from torch.autograd import grad

from pinc.model import Params, StaticLossArgs, beta_softplus, compute_loss, compute_variables, mlp_forward


def bumpft(val, epsilon=0.1):
    return 1 - torch.tanh(val / epsilon) ** 2


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs, inputs=inputs, grad_outputs=d_points, create_graph=True, retain_graph=True, only_inputs=True
    )[0][:, -3:]
    return points_grad  # noqa: RET504


# Slightly modified code from https://github.com/Yebbi/PINC/blob/main/model/network.py#L18
class ImplicitNet_PINC(nn.Module):
    def __init__(
        self,
        dims,
        skip_in=(),
        init_type="geo_relu",
        radius_init=1,
        beta=100.0,
    ):
        super().__init__()
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.init_type = init_type

        for layer in range(self.num_layers - 1):
            if layer + 1 in skip_in:  # noqa: SIM108
                out_dim = dims[layer + 1] - 3
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)
            # if true preform geometric initialization
            if self.init_type == "geo_relu":
                if layer == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -0.1 * radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

    def forward(self, inp):
        output = inp
        for layer in range(self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                output = torch.cat([output, inp], -1) / np.sqrt(2)

            output = lin(output)

            if layer < self.num_layers - 2:
                output = self.activation(output)

        SDF = output[:, 0]

        grad = gradient(inp, SDF)

        grad_f1 = gradient(inp, output[:, 1:2])
        grad_f2 = gradient(inp, output[:, 2:3])
        grad_f3 = gradient(inp, output[:, 3:4])
        A1 = grad_f3[:, 1:2] - grad_f2[:, 2:3] - inp[:, 0:1] / 3.0
        A2 = grad_f1[:, 2:3] - grad_f3[:, 0:1] - inp[:, 1:2] / 3.0
        A3 = grad_f2[:, 0:1] - grad_f1[:, 1:2] - inp[:, 2:3] / 3.0
        grad_SDF = torch.cat([A1, A2, A3], dim=-1)
        grad_SDF = grad_SDF / (torch.linalg.norm(grad_SDF, dim=1, keepdim=True) + 1e-10)

        aug_grad = output[:, 4:7] / (torch.nn.ReLU()(torch.linalg.norm(output[:, 4:7], dim=1, keepdim=True) - 1) + 1)

        return {
            "output": output,
            "SDF_pred": SDF,
            "grad": grad,
            "grad_pred": grad_SDF,
            "auggrad_pred": aug_grad,
        }

    def compute_loss(self, mnfld_pnts, nonmnfld_pnts, epsilon, regularizer_coord):
        """Slightly modified code from https://github.com/Yebbi/PINC/blob/main/reconstruction/run.py#L65"""
        # Forward pass
        mnfld_outputs = self(mnfld_pnts)
        nonmnfld_outputs = self(nonmnfld_pnts)

        mnfld_pred = mnfld_outputs["SDF_pred"]
        mnfld_G = mnfld_outputs["grad_pred"]
        mnfld_G_tilde = mnfld_outputs["auggrad_pred"]
        nonmnfld_pred = nonmnfld_outputs["SDF_pred"]
        nonmnfld_G = nonmnfld_outputs["grad_pred"]
        nonmnfld_G_tilde = nonmnfld_outputs["auggrad_pred"]

        # Compute grad
        mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
        nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)

        # Manifold loss
        mnfld_loss = (mnfld_pred.abs()).mean()

        # Gradient Matching loss (L2 penalty term for the constraint G=\nabla u)
        grad_loss = (
            ((nonmnfld_grad - nonmnfld_G).norm(2, dim=-1) ** 2).mean() * nonmnfld_pnts.shape[0]
            + ((mnfld_grad - mnfld_G).norm(2, dim=-1) ** 2).mean() * mnfld_pnts.shape[0]
        ) / (nonmnfld_pnts.shape[0] + mnfld_pnts.shape[0])

        # Minimal Area loss
        area_loss = (
            (bumpft(nonmnfld_pred, epsilon=epsilon) * ((nonmnfld_grad).norm(2, dim=-1))).mean() * nonmnfld_pnts.shape[0]
            + (bumpft(mnfld_pred, epsilon=epsilon) * ((mnfld_grad).norm(2, dim=-1))).mean() * mnfld_pnts.shape[0]
        ) / (nonmnfld_pnts.shape[0] + mnfld_pnts.shape[0])

        # Curl loss
        H1 = gradient(mnfld_pnts, mnfld_G_tilde[:, 0])
        H2 = gradient(mnfld_pnts, mnfld_G_tilde[:, 1])
        H3 = gradient(mnfld_pnts, mnfld_G_tilde[:, 2])
        curlG_x = H3[:, 1] - H2[:, 2]
        curlG_y = H1[:, 2] - H3[:, 0]
        curlG_z = H2[:, 0] - H1[:, 1]
        curl_loss_mnfld = (curlG_x**2 + curlG_y**2 + curlG_z**2).mean()
        del H1, H2, H3, curlG_x, curlG_y, curlG_z

        H1 = gradient(nonmnfld_pnts, nonmnfld_G_tilde[:, 0])
        H2 = gradient(nonmnfld_pnts, nonmnfld_G_tilde[:, 1])
        H3 = gradient(nonmnfld_pnts, nonmnfld_G_tilde[:, 2])
        curlG_x = H3[:, 1] - H2[:, 2]
        curlG_y = H1[:, 2] - H3[:, 0]
        curlG_z = H2[:, 0] - H1[:, 1]
        curl_loss_nonmnfld = (curlG_x**2 + curlG_y**2 + curlG_z**2).mean()
        del H1, H2, H3, curlG_x, curlG_y, curlG_z

        curl_loss = (curl_loss_mnfld * mnfld_pnts.shape[0] + curl_loss_nonmnfld * nonmnfld_pnts.shape[0]) / (
            nonmnfld_pnts.shape[0] + mnfld_pnts.shape[0]
        )

        # Matching two auxiliary variables
        G_matching = (
            ((nonmnfld_G_tilde - nonmnfld_G).norm(2, dim=-1) ** 2).mean() * nonmnfld_pnts.shape[0]
            + ((mnfld_G_tilde - mnfld_G).norm(2, dim=-1) ** 2).mean() * mnfld_pnts.shape[0]
        ) / (nonmnfld_pnts.shape[0] + mnfld_pnts.shape[0])

        loss = (
            mnfld_loss
            + regularizer_coord[0] * grad_loss
            + regularizer_coord[1] * G_matching
            + regularizer_coord[2] * curl_loss
            + regularizer_coord[3] * area_loss
        )
        return {
            "loss": loss,
            "mnfld_loss": mnfld_loss,
            "grad_loss": regularizer_coord[0] * grad_loss,
            "G_matching": regularizer_coord[1] * G_matching,
            "curl_loss": regularizer_coord[2] * curl_loss,
            "area_loss": regularizer_coord[3] * area_loss,
        }


# %%
dims = [3] + [512] * 7 + [7]
skip_layers = [4]
set_random_seed(1234)

# %% The published torch model
torch_model = ImplicitNet_PINC(dims, skip_in=skip_layers)

# %% Extract the params from the model
params = []
for layer in range(torch_model.num_layers - 1):
    layer = getattr(torch_model, "lin" + str(layer))
    params.append((layer.weight.detach().numpy(), layer.bias.detach().numpy()))

# %% Create a dummy input
x = np.random.rand(3).astype(np.float32)  # noqa: NPY002

# %% Forward pass of torch model
torch_x = torch.tensor(x).reshape(1, -1)
torch_x.requires_grad_()
torch_out = torch_model(torch_x)
torch_out_cpu = jax.tree_map(lambda x: x.detach().numpy(), torch_out)  # put on cpu in numpy

# %% Compute output using jax
activation = partial(beta_softplus, beta=100.0)
jax_out = mlp_forward(params, x, activation=activation, skip_layers=skip_layers)

# %% Check if the outputs are the same
assert np.allclose(jax_out, torch_out_cpu["output"])
assert jax_out.shape == torch_out_cpu["output"][0].shape

# %% Compute the gradients
F = lambda x: x / 3
sdf, grad_sdf, G, G_tilde, curl_G_tilde = compute_variables(params, x, activation=activation, F=F, skip_layers=skip_layers)

# %%
# mapping:
# sdf -> SDF_pred
# G -> grad_pred
# G_tilde -> auggrad_pred
assert np.allclose(sdf, torch_out_cpu["SDF_pred"])
assert np.allclose(G, torch_out_cpu["grad_pred"])
assert np.allclose(G_tilde, torch_out_cpu["auggrad_pred"])

# %%

epsilon = 0.1
n_points = 16384
loss_weights = np.array([0.1, 1e-4, 5e-4, 0.1])
boundary_points = np.random.rand(n_points, 3).astype(np.float32)  # noqa: NPY002
sample_points = np.random.rand(n_points // 8, 3).astype(np.float32)  # noqa: NPY002

# %%
torch_boundary_points = torch.tensor(boundary_points)
torch_sample_points = torch.tensor(sample_points)
torch_boundary_points.requires_grad_()
torch_sample_points.requires_grad_()
torch_losses = torch_model.compute_loss(torch_boundary_points, torch_sample_points, epsilon, loss_weights)
torch_losses_cpu = jax.tree_map(lambda x: x.detach().numpy(), torch_losses)

# %%
static = StaticLossArgs(
    activation=activation,
    F=F,
    skip_layers=skip_layers,
    loss_weights=loss_weights,
    epsilon=epsilon,
)


def batch_loss(params: Params, boundary_points: Array, sample_points: Array):
    loss_sdf, boundary_loss_terms = vmap(partial(compute_loss, params, static=static))(boundary_points)
    _, sample_loss_terms = vmap(partial(compute_loss, params, static=static))(sample_points)
    n_points = len(boundary_points) + len(sample_points)
    loss_terms_sum = (boundary_loss_terms.sum(axis=0) + sample_loss_terms.sum(axis=0)) / n_points
    loss_sdf_mean = loss_sdf.mean()
    return loss_sdf_mean + loss_terms_sum.sum(), (loss_sdf_mean, loss_terms_sum)


jax_losses = batch_loss(params, boundary_points, sample_points)
loss, (loss_sdf, loss_terms_sum) = jax_losses

# %%
# mapping:
# loss -> loss
# loss_sdf -> mnfld_loss
# loss_terms_sum[0] -> grad_loss
# loss_terms_sum[1] -> G_matching
# loss_terms_sum[2] -> curl_loss
# loss_terms_sum[3] -> area_loss
assert np.allclose(loss, torch_losses_cpu["loss"])
assert np.allclose(loss_sdf, torch_losses_cpu["mnfld_loss"])
assert np.allclose(loss_terms_sum[0], torch_losses_cpu["grad_loss"])
assert np.allclose(loss_terms_sum[1], torch_losses_cpu["G_matching"])
assert np.allclose(loss_terms_sum[2], torch_losses_cpu["curl_loss"])
assert np.allclose(loss_terms_sum[3], torch_losses_cpu["area_loss"])

# %%
