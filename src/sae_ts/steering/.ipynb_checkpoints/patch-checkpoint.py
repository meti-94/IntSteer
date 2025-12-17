from typing import List, Callable, Union, Optional, Literal

from functools import partial
from tqdm import tqdm
import json
import sys


import torch
from torch.utils.data import DataLoader
import einops
import math

from datasets import load_dataset

# from .evals_utils import evaluate_completions

from transformer_lens import HookedTransformer
from transformer_lens.utilities import devices
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
import transformer_lens.utils as tutils



# def patch_resid(resid, hook, steering, scale=1):
#     resid[:, :, :] = resid[:, :, :] + steering * scale
#     return resid

# def patch_resid(resid, hook, steering, scale, storage):
#     storage['before'].append(resid[:, -1, :].clone())
#     resid[:, :, :] = resid[:, :, :] + steering * scale
#     storage['after'].append(resid[:, -1, :].clone())
#     return resid






# def patch_resid(resid, hook, steering, scale, storage):
#     """
#     Slerp-based steering: rotate each residual vector toward `steering`
#     with strength `scale` in [0, 1], while preserving the original ||resid||.

#     resid:    (..., d) tensor, typically [batch, seq, d_model]
#     steering: (d) tensor or broadcastable to resid's shape
#     scale:    float or tensor in [0, 1]; alpha for the slerp
#     """
#     storage['before'].append(resid[:, -1, :].clone())
#     eps = 1e-8
#     device = resid.device
#     dtype = resid.dtype

#     # Prepare steering to match resid's shape
#     y = steering.to(device=device, dtype=dtype)
#     # If y is a single direction (1D of size d), unsqueeze/broadcast to resid
#     while y.dim() < resid.dim():
#         y = y.unsqueeze(0)
#     if y.shape != resid.shape:
#         y = y.expand_as(resid)

#     # Compute per-vector norms and unit directions
#     x_norm = torch.linalg.norm(resid, dim=-1, keepdim=True)
#     y_norm = torch.linalg.norm(y,     dim=-1, keepdim=True)

#     # Avoid division by zero; if a norm is ~0, fall back to a safe default
#     xhat = resid / (x_norm + eps)
#     # If steering has zero norm somewhere, use xhat there (no steering effect)
#     yhat = torch.where(y_norm > eps, y / (y_norm + eps), xhat)

#     # Alpha (steering strength) as a broadcastable tensor
#     if isinstance(scale, (float, int)):
#         alpha = torch.tensor(scale, device=device, dtype=dtype)
#     else:
#         alpha = scale.to(device=device, dtype=dtype)
#     while alpha.dim() < resid.dim():
#         # Make alpha broadcast along the last dimension
#         alpha = alpha.unsqueeze(-1)
#     # Clamp for numerical hygiene
#     alpha = torch.clamp(alpha, 0.0, 1.0)

#     # Slerp core: angle and sinus terms
#     dot = (xhat * yhat).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
#     theta = torch.acos(dot)
#     sin_theta = torch.sin(theta)

#     # Masks for edge cases
#     small_angle = sin_theta < 1e-6          # xhat ~ yhat
#     opposite_dir = dot < -0.9999            # xhat ~ -yhat

#     # Standard slerp where sin(theta) is safe
#     A = torch.sin((1.0 - alpha) * theta) / (sin_theta + eps)
#     B = torch.sin(alpha * theta) / (sin_theta + eps)
#     zhat_slerp = A * xhat + B * yhat  # unit direction on the sphere

#     # Fallback when angle is tiny: normalized linear blend
#     zhat_lerp = torch.nn.functional.normalize(
#         (1.0 - alpha) * xhat + alpha * yhat, dim=-1, eps=eps
#     )

#     # Special handling for near-opposite directions:
#     # rotate in the plane spanned by xhat and the orthogonal component of yhat
#     v = yhat - dot * xhat                       # component orthogonal to xhat
#     v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)
#     # If v is tiny (degenerate), build a deterministic orthonormal complement
#     # via a basis vector least aligned with xhat:
#     idx = torch.argmin(torch.abs(xhat), dim=-1, keepdim=True)  # (..., 1)
#     e = torch.zeros_like(xhat).scatter(-1, idx, 1.0)           # unit basis vector
#     # dot(e, xhat) = gather(xhat, idx)
#     dot_e = torch.gather(xhat, -1, idx)                        # (..., 1)
#     v_alt = e - dot_e * xhat
#     vhat = torch.where(v_norm > eps, v / (v_norm + eps),
#                        torch.nn.functional.normalize(v_alt, dim=-1, eps=eps))
#     # Rotate by alpha * pi in this plane
#     zhat_opposite = torch.cos(alpha * math.pi) * xhat + torch.sin(alpha * math.pi) * vhat

#     # Combine cases
#     # Note: torch.where preserves shape; masks are (..., 1) so they broadcast over last dim.
#     zhat = torch.where(opposite_dir, zhat_opposite,
#            torch.where(small_angle, zhat_lerp, zhat_slerp))

#     # Restore original per-vector norms to preserve magnitude
#     z = x_norm * zhat

#     # In-place update to match your original pattern
#     resid[:, :, :] = z
#     storage['after'].append(resid[:, -1, :].clone())
#     return resid



def patch_resid(resid, hook, steering, scale, storage):
    """
    Tangent exponential-map steering:
    Rotate each residual vector toward the component of `steering` orthogonal to it,
    by an angle phi = scale * theta, where theta is the angle between directions.
    Preserves the original per-vector norms of `resid`.

    resid:    (..., d) tensor, typically [batch, seq, d_model]
    steering: (d) tensor or broadcastable to resid's shape
    scale:    float or tensor in [0, 1]; acts as a fraction of the angle to the steering direction
    """
    storage['before'].append(resid[:, -1, :].clone())
    eps = 1e-8
    device = resid.device
    dtype = resid.dtype

    # Prepare steering to match resid's shape
    y = steering.to(device=device, dtype=dtype)
    while y.dim() < resid.dim():
        y = y.unsqueeze(0)
    if y.shape != resid.shape:
        y = y.expand_as(resid)

    # Norms and unit directions
    x_norm = torch.linalg.norm(resid, dim=-1, keepdim=True)
    y_norm = torch.linalg.norm(y,     dim=-1, keepdim=True)
    xhat = resid / (x_norm + eps)
    yhat = torch.where(y_norm > eps, y / (y_norm + eps), xhat)  # if steering ~0, do nothing

    # Angle between directions and its sine
    dot = (xhat * yhat).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)  # (..., 1)

    # Orthogonal component of yhat relative to xhat
    v = yhat - dot * xhat
    v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)

    # Build a deterministic orthonormal complement if v is tiny (colinear case)
    # Choose basis axis least aligned with xhat
    idx = torch.argmin(torch.abs(xhat), dim=-1, keepdim=True)   # (..., 1)
    e = torch.zeros_like(xhat).scatter(-1, idx, 1.0)            # unit basis vector
    dot_e = torch.gather(xhat, -1, idx)                         # (..., 1)
    v_alt = e - dot_e * xhat
    vhat = torch.where(v_norm > eps,
                       v / (v_norm + eps),
                       torch.nn.functional.normalize(v_alt, dim=-1, eps=eps))

    # Steering strength: phi = alpha * theta
    if isinstance(scale, (float, int)):
        alpha = torch.tensor(scale, device=device, dtype=dtype)
    else:
        alpha = scale.to(device=device, dtype=dtype)
    # Broadcast alpha to match all but the last dim
    while alpha.dim() < resid.dim():
        alpha = alpha.unsqueeze(-1)
    alpha = torch.clamp(alpha, 0.0, 1.0)
    phi = alpha * theta  # radians

    # Exponential-map rotation in the 2D plane spanned by xhat and vhat
    zhat = torch.cos(phi) * xhat + torch.sin(phi) * vhat

    # Restore original norm to preserve magnitude
    z = x_norm * zhat

    # In-place update
    resid[:, :, :] = z
    storage['after'].append(resid[:, -1, :].clone())
    return resid
