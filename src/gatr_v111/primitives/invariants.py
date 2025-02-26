# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# from functools import lru_cache

import torch
import torch.linalg

from src.gatr_v111.primitives.bilinear import _load_bilinear_basis
from src.gatr_v111.primitives.linear import _compute_reversal, grade_project
from src.gatr_v111.utils.einsum import cached_einsum


# @lru_cache()
def compute_inner_product_mask(gp, device=torch.device("cpu")) -> torch.Tensor:
    """Constructs a bool array for the inner product calculation.

    The inner product of MVs is <~x y>_0, i.e. take the grade-0 component of the geometric
    product of the reverse of x with y.
    Both the scalar component of the GP, and the reversal matrix, are diagonal.
    Their product is 0 for basis elements involving e0, and 1 elsewhere, i.e.
    IP = [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
    for dim order '', 'e0', 'e1', 'e2', 'e3', 'e01', 'e02', 'e03', 'e12', 'e13', 'e23',
                  'e012', 'e013', 'e023', 'e123', 'e0123'

    Parameters
    ----------
    device : torch.device
        Device

    Returns
    -------
    ip_mask : torch.Tensor with shape (16,)
        Inner product mask
    """
    # gp = _load_bilinear_basis("gp", device=device, dtype=torch.float32)
    inner_product_mask = torch.diag(gp[0]) * _compute_reversal(device=device, dtype=torch.float32)
    return inner_product_mask.bool()


def inner_product(colums_take, x: torch.Tensor, y: torch.Tensor, channel_sum: bool = False) -> torch.Tensor:
    """Computes the inner product of multivectors f(x,y) = <x, y> = <~x y>_0.

    In addition to summing over the 16 multivector dimensions, this function also sums
    over an additional channel dimension if channel_sum == True.

    Equal to `geometric_product(reverse(x), y)[..., [0]]` (but faster).

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16) or (..., channels, 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor with shape (..., 16) or (..., channels, 16)
        Second input multivector. Batch dimensions must be broadcastable between x and y.
    channel_sum: bool
        Whether to sum over the second-to-last axis (channels)

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
        Result. Batch dimensions are result of broadcasting between x and y.
    """
    # changed 26.09 23.53  ######################################################
    # mask = compute_inner_product_mask(gp, device=x.device)
    # print("mask", mask)
    # print("mask", mask.shape)
    # print("x before mask", x.shape)
    # x = x[..., compute_inner_product_mask(gp, device=x.device)]
    # y = y[..., compute_inner_product_mask(gp, device=x.device)]

    # print(colums_take)
    # print("x after mask", x.shape)
    x = torch.index_select(
            x, -1, colums_take.long().to(x.device)
        )
    y = torch.index_select(
            y, -1, colums_take.long().to(y.device)
        )
    


    ######################################################
    # if channel_sum:
    #     outputs = cached_einsum("... c i, ... c i -> ...", x, y)
    # else:
    #     outputs = cached_einsum("... i, ... i -> ...", x, y)
    # if channel_sum:
    #     outputs = torch.sum(torch.sum(torch.mul(x, y), dim=-1), dim=-1)
    #     # outputs = torch.einsum("... c i, ... c i -> ...", x, y)
    # else:
    if channel_sum:
        outputs = torch.sum(torch.sum(torch.mul(x, y), dim=-1), dim=-1)
        # outputs = torch.einsum("... c i, ... c i -> ...", x, y)
    else:
        outputs = torch.sum(torch.mul(x, y), dim=-1)
        # outputs = torch.einsum("... i, ... i -> ...", x, y)
        # outputs = torch.einsum("... i, ... i -> ...", x, y)
    # We want the output to have shape (..., 1)
    outputs = outputs.unsqueeze(-1)

    return outputs


def norm(x: torch.Tensor) -> torch.Tensor:
    """Computes the GA norm of an input multivector.

    Equal to sqrt(inner_product(x, x)).

    NOTE: this primitive is not used widely in our architectures.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
        Geometric algebra norm of x.
    """

    return torch.sqrt(torch.clamp(inner_product(x, x), 0.0))


def pin_invariants(x: torch.Tensor) -> torch.Tensor:
    """Computes five invariants from multivectors: scalar component, norms of the four other grades.

    NOTE: this primitive is not used widely in our architectures.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 5)
        Invariants computed from input multivectors
    """

    # Project to grades
    projections = grade_project(x)  # (..., 5, 16)

    # Compute norms
    squared_norms = inner_product(projections, projections)[..., 0]  # (..., 5)
    norms = torch.sqrt(torch.clamp(squared_norms, 0.0))

    # Outputs: scalar component of input and norms of four other grades
    return torch.cat((x[..., [0]], norms[..., 1:]), dim=-1)  # (..., 5)
