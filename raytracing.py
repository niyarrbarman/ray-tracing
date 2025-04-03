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
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    NR = rays.size(0)
    NS = segments.size(0)

    # Get just the x and y coordinates
    rays = rays[..., :2]
    segments = segments[..., :2]

    # Repeat rays and segments so that we can compuate the intersection of every (ray, segment) pair
    rays = einops.repeat(rays, "nrays p d -> nrays nsegments p d", nsegments=NS)
    segments = einops.repeat(segments, "nsegments p d -> nrays nsegments p d", nrays=NR)

    # Each element of `rays` is [[Ox, Oy], [Dx, Dy]]
    O = rays[:, :, 0]
    D = rays[:, :, 1]
    assert O.shape == (NR, NS, 2)

    # Each element of `segments` is [[L1x, L1y], [L2x, L2y]]
    L_1 = segments[:, :, 0]
    L_2 = segments[:, :, 1]
    assert L_1.shape == (NR, NS, 2)

    # Define matrix on left hand side of equation
    mat = t.stack([D, L_1 - L_2], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    assert is_singular.shape == (NR, NS)
    mat[is_singular] = t.eye(2)

    # Define vector on the right hand side of equation
    vec = L_1 - O

    # Solve equation, get results
    sol = t.linalg.solve(mat, vec)
    u = sol[..., 0]
    v = sol[..., 1]

    # Return boolean of (matrix is nonsingular, and solution is in correct range implying intersection)
    return ((u >= 0) & (v >= 0) & (v <= 1) & ~is_singular).any(dim=-1)

tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %% 2D Rays
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    rays = t.zeros((num_pixels_y * num_pixels_z, 2, 3))
    y_rays = t.linspace(-y_limit, y_limit, num_pixels_y)
    z_rays = t.linspace(-z_limit, z_limit, num_pixels_y)
    rays[:, 1, 0] = 1
    rays[:, 1, 1] = einops.repeat(y_rays, 'y -> (y z)', z=num_pixels_z)
    rays[:, 1, 2] = einops.repeat(z_rays, 'z -> (y z)', y=num_pixels_y)
    return rays

rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)

# %% Triangles
one_triangle = t.tensor([[0, 0, 0], [4, 0.5, 0], [2, 3, 0]])
A, B, C = one_triangle
x, y, z = one_triangle.T

fig: go.FigureWidget = setup_widget_fig_triangle(x, y, z)
display(fig)


@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def update(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.update_traces({"x": [P[0]], "y": [P[1]]}, 2)
# %% Triangle-Ray Intesection
Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    delta_1 = B - A
    delta_2 = C - A
    C1 = O - A


    matrix = t.stack([-D, delta_1, delta_2], dim=1)
    
    s, u, v = t.linalg.solve(matrix, C1)

    return (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)

tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %% Single-Tringle Rendering

def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"], triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    NR = rays.size(0)

    A, B, C = einops.repeat(triangle, "pts dims -> pts nrays dims", nrays=NR)
    O, D = rays.unbind(dim=1)

    mat: Float[Tensor, "NR 3 3"] = t.stack([-D, B - A, C - A], dim=-1)
    dets: Float[Tensor, "NR"] = t.linalg.det(mat)
    is_singular = t.isclose(dets, t.tensor(0.0))
    mat[is_singular] = t.eye(3)

    vec = O-A
    
    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular

A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 10
y_limit = z_limit = 0.5

test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Ray-Traced Triangle")

# %% Mesh Rendering

def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"], triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    NR = rays.size(0)
    NT = triangles.size(0)
    
    triangles = einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=NR)
    A, B, C = triangles  

    rays = einops.repeat(rays, "NR pts dims -> pts NR NT dims", NT=NT)
    O, D = rays

    mat: Float[Tensor, "NR NT 3 3"] = t.stack([-D, B - A, C - A], dim=-1)
    dets: Float[Tensor, "NR NT"] = t.linalg.det(mat)
    is_singular = t.isclose(dets, t.tensor(0.0))
    mat[is_singular] = t.eye(3)

    vec = O-A
    sol: Float[Tensor, "NR NT 3"] = t.linalg.solve(mat, vec)
    
    s, u, v = sol.unbind(dim=-1)
    intersects = (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular
    s[~intersects] = t.tensor(float("inf"))

    return einops.reduce(s, "NR NT -> NR", "min") 


num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

triangles = t.load(os.path.join(section_dir, "pikachu.pt"), weights_only=True)
rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])

dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()

# %%
