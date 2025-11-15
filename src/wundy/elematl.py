from typing import Callable, Protocol, Any, Optional
import numpy as np
from numpy.typing import NDArray

# ---------- Materials ----------

class Material1D(Protocol):
    """Protocol for 1D small-strain materials."""
    def stress(self, strain: float) -> float: ...
    def tangent(self, strain: float) -> float: ...
    # Optional convenience attribute for density (used by GRAV body load)
    rho: float | None

class LinearElastic1D:
    """
    σ = E (ε - α ΔT);  C_tan = E
    Set alpha=0 or dT=0 to ignore thermal strain.
    """
    def __init__(self, E: float, alpha: float = 0.0, dT: float = 0.0, rho: Optional[float] = None):
        self.E = float(E)
        self.alpha = float(alpha)
        self.dT = float(dT)
        self.rho = None if rho is None else float(rho)

    def stress(self, strain: float) -> float:
        return self.E * (strain - self.alpha * self.dT)

    def tangent(self, strain: float) -> float:
        return self.E

def make_material(material_spec: dict[str, Any]) -> Material1D:
    """
    Adapter: your input 'materials' dict -> a material object with stress()/tangent().
    Expected schema (current code already uses this layout):
        {
          "type": "linear_elastic",
          "parameters": {"E": ..., "alpha": 0.0, "dT": 0.0},
          "density": ... (optional)
        }
    """
    mtype = material_spec.get("type", "linear_elastic").lower()
    params = material_spec.get("parameters", {})
    rho = material_spec.get("density", None)

    if mtype in {"linear", "elastic", "linear_elastic", "linear-elastic"}:
        return LinearElastic1D(
            E=params["E"],
            alpha=params.get("alpha", 0.0),
            dT=params.get("dT", 0.0),
            rho=rho,
        )
    raise NotImplementedError(f"Material type {mtype!r} not supported.")


# ---------- Quadrature utilities ----------

# 2-point Gauss on [-1, 1]
_GAUSS_XI_2 = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
_GAUSS_W_2  = np.array([1.0, 1.0])

def _shape_lin(xi: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Linear 2-node bar on parent [-1,1]."""
    N = np.array([0.5 * (1 - xi), 0.5 * (1 + xi)], dtype=float)
    dN_dxi = np.array([-0.5, 0.5], dtype=float)
    return N, dN_dxi

def _kinematics_1d(xe: NDArray[float], ue: Optional[NDArray[float]], xi: float
                   ) -> tuple[float, float, NDArray[float], float]:
    """
    Return (x, J, B, strain) at Gauss point xi for a 2-node linear bar element.
    xe: shape (2,) physical coordinates
    ue: shape (2,) nodal displacements (optional for stiffness)
    """
    N, dN_dxi = _shape_lin(xi)
    h = float(xe[1] - xe[0])
    if np.isclose(h, 0.0):
        raise ValueError("Zero-length element.")
    J = h / 2.0
    dN_dx = dN_dxi / J
    B = dN_dx  # for 1D bar, B = [dN1/dx, dN2/dx]
    x = float(N @ xe)
    strain = float(B @ ue) if ue is not None else 0.0
    return x, J, B, strain

def _as_area_func(A: float | Callable[[float], float]) -> Callable[[float], float]:
    if callable(A):
        return A
    a = float(A)
    return lambda x: a

# ---------- Element responses (Gauss-based) ----------

def element_stiffness_bar1d(
    xe: NDArray[float],                      # shape (2,)
    A: float | Callable[[float], float],     # constant or A(x)
    material: Material1D,
    n_gauss: int = 2,
) -> NDArray[float]:
    """Ke = ∑ B^T C B A(x) J w"""
    if n_gauss != 2:
        raise NotImplementedError("Only 2-pt Gauss implemented for now.")
    Ke = np.zeros((2, 2), dtype=float)
    A_of_x = _as_area_func(A)
    for xi, w in zip(_GAUSS_XI_2, _GAUSS_W_2):
        x, J, B, _ = _kinematics_1d(xe, ue=None, xi=xi)
        C = material.tangent(0.0)  # linear elastic -> constant
        Ke += np.outer(B, B) * C * A_of_x(x) * J * w
    return Ke

def element_internal_force_bar1d(
    xe: NDArray[float],                      # shape (2,)
    ue: NDArray[float],                      # shape (2,)
    A: float | Callable[[float], float],
    material: Material1D,
    n_gauss: int = 2,
) -> NDArray[float]:
    """f_int = ∑ B^T σ A(x) J w"""
    if n_gauss != 2:
        raise NotImplementedError("Only 2-pt Gauss implemented for now.")
    fint = np.zeros(2, dtype=float)
    A_of_x = _as_area_func(A)
    for xi, w in zip(_GAUSS_XI_2, _GAUSS_W_2):
        x, J, B, strain = _kinematics_1d(xe, ue, xi)
        sigma = material.stress(strain)
        fint += B * sigma * A_of_x(x) * J * w
    return fint

def element_external_body_bar1d(
    xe: NDArray[float],                          # shape (2,)
    q_of_x: float | Callable[[float], float],    # force/length; constant or function of x
    n_gauss: int = 2,
) -> NDArray[float]:
    """Consistent nodal load: f_ext = ∑ N q(x) J w"""
    if n_gauss != 2:
        raise NotImplementedError("Only 2-pt Gauss implemented for now.")
    if callable(q_of_x):
        qx = q_of_x
    else:
        q_const = float(q_of_x)
        qx = lambda x: q_const

    fext = np.zeros(2, dtype=float)
    for xi, w in zip(_GAUSS_XI_2, _GAUSS_W_2):
        N, dN_dxi = _shape_lin(xi)
        h = float(xe[1] - xe[0])
        J = h / 2.0
        x = float(N @ xe)
        fext += N * qx(x) * J * w
    return fext
