from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from .elematl import make_material


def _sorted_by_x(coords: NDArray[np.float64]) -> NDArray[np.int64]:
    x = np.asarray(coords[:, 0], dtype=float)
    return np.argsort(x)


def _element_stresses(
    coords: NDArray[np.float64],
    blocks: list[dict[str, Any]],
    materials: dict[str, Any],
    dofs: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Returns (x_plot, sigma_plot) where sigma is piecewise-constant per element.
    Uses a NaN separator between elements so matplotlib breaks the line segments.
    """
    x_segs: list[float] = []
    s_segs: list[float] = []

    for block in blocks:
        mat = make_material(materials[block["material"]])

        for conn in block["connect"]:
            n1, n2 = int(conn[0]), int(conn[1])

            x1 = float(coords[n1, 0])
            x2 = float(coords[n2, 0])
            u1 = float(dofs[n1])
            u2 = float(dofs[n2])

            h = x2 - x1
            if np.isclose(h, 0.0):
                raise ValueError("Zero-length element encountered while computing stress.")

            # small strain in 1D linear kinematics
            strain = (u2 - u1) / h
            sigma = float(mat.stress(strain))

            # ensure left-to-right segment for plotting
            xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)

            x_segs.extend([xa, xb, np.nan])
            s_segs.extend([sigma, sigma, np.nan])

    return np.asarray(x_segs, dtype=float), np.asarray(s_segs, dtype=float)


def plot_displacement(
    coords: NDArray[np.float64],
    dofs: NDArray[np.float64],
    *,
    ax: plt.Axes | None = None,
    title: str = "Displacement (piecewise-linear)",
) -> plt.Axes:
    """
    Piecewise-linear displacement plot: connect nodal values in x-order.
    """
    x = np.asarray(coords[:, 0], dtype=float)
    u = np.asarray(dofs, dtype=float)

    order = np.argsort(x)
    xs = x[order]
    us = u[order]

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(xs, us, marker="o")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(title)
    ax.grid(True)
    return ax


def plot_stress(
    coords: NDArray[np.float64],
    blocks: list[dict[str, Any]],
    materials: dict[str, Any],
    dofs: NDArray[np.float64],
    *,
    ax: plt.Axes | None = None,
    title: str = "Stress (piecewise-constant per element)",
) -> plt.Axes:
    """
    Piecewise-constant stress plot (step-like per element).
    Stress is computed from element strain using the block material model.
    """
    xp, sp = _element_stresses(coords, blocks, materials, dofs)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(xp, sp, marker=None)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\sigma$")
    ax.set_title(title)
    ax.grid(True)
    return ax


def plot_bar1d_results(
    coords: NDArray[np.float64],
    blocks: list[dict[str, Any]],
    materials: dict[str, Any],
    soln: dict[str, Any],
    *,
    show: bool = True,
    save_prefix: str | None = None,
) -> None:
    """
    Convenience wrapper: displacement + stress figures.
    If save_prefix is provided, saves:
      {save_prefix}_disp.png and {save_prefix}_stress.png
    """
    dofs = np.asarray(soln["dofs"], dtype=float)

    ax1 = plot_displacement(coords, dofs)
    fig1 = ax1.figure

    ax2 = plot_stress(coords, blocks, materials, dofs)
    fig2 = ax2.figure

    if save_prefix:
        fig1.savefig(f"{save_prefix}_disp.png", dpi=200, bbox_inches="tight")
        fig2.savefig(f"{save_prefix}_stress.png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()
