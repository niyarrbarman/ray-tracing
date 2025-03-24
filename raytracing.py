# %% Setup
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = "../ARENA_3.0"
exercises_dir = os.path.join(root_dir, chapter, "exercises")
section_dir = os.path.join(exercises_dir, section)
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests # type: ignore
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle # type: ignore
from plotly_utils import imshow # type: ignore


MAIN = __name__ == "__main__"

# %% 1D Image Rendering
def make_rays_1d(num_pixels: int, y_limit:float) -> Tensor:
    rays = t.zeros((num_pixels, 2,3))
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays

rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)
    
# %% Ray-Segment Intersection
def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    O,D = ray[:,:2]
    L1,L2 = segment[:,:2]
    
    delta = L2 - L1
    C = L1 - O

    matrix = t.stack([D, -delta], dim=-1)
    
    try:
        t1, t2 = t.linalg.solve(matrix, C)
    except RuntimeError:
        return False
    
    return t1.item() >= 0 and 0 <= t2.item() <= 1


tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

# %% Batched Ray-Segment Intersection
def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    """
    For each ray, check if it intersects any of the segments.
    """
    
    NR = rays.shape[0]
    NS = segments.shape[0]

    # Just the x and y coordinates
    rays = rays[..., : 2]
    segments = segments[..., : 2]

    rays = einops.repeat(rays, 'nr p d -> nr ns p d', ns=NS)
    segments = einops.repeat(segments, 'ns p d -> nr ns p d', nr=NR)

    O = rays[:, :, 0]
    D = rays[:, :, 1]
    assert O.shape == (NR, NS, 2)

    L1 = segments[:, :, 0]
    L2 = segments[:, :, 1]
    assert L1.shape == (NR, NS, 2)

    delta = L2 - L1
    C = L1 - O

    matrix = t.stack([D, -delta], dim=-1)
    dets = t.linalg.det(matrix)
    is_singular = t.isclose(dets, t.tensor(0.0))
    assert is_singular.shape == (NR, NS)
    matrix[is_singular] = t.eye(2)

    sol = t.linalg.solve(matrix, C)
    t1, t2 = sol[..., 0], sol[..., 1]

    print(t1.shape, t2.shape, is_singular.shape)

    return (t1 >= 0) & (t2 >= 0) & (t2 <= 1) & (~is_singular).any(dim=-1)

tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %%
