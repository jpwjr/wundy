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

class NeoHookean1D:
    """
    1D Neo-Hookean material.

    For pure 1D bars, volumetric stiffness K does not appear.
    We therefore default K = 0 unless explicitly provided.

        W(λ) = 0.5 μ (λ^2 - 1 - 2 ln λ)      [K=0 for 1D]

        σ(ε) = μ (λ - 1/λ)
        C_tan(ε) = μ (1 + 1/λ²)

    where λ = 1 + ε > 0.
    """

    def __init__(self, mu: float, K: float = 0.0, rho: float | None = None, nu = None):
        self.mu = float(mu)
        self.K = float(K)  # usually 0 for 1D
        self.rho = None if rho is None else float(rho)
        self.nu = nu

    def _stretch(self, strain: float) -> float:
        lam = 1.0 + float(strain)
        if lam <= 0.0:
            raise ValueError(
                f"NeoHookean1D: invalid stretch λ={lam:.6g} from strain {strain:.6g}; "
                "λ must be > 0."
            )
        return lam

    def stress(self, strain: float) -> float:
        lam = self._stretch(strain)
        mu = self.mu
        K = self.K
        return mu * (lam - 1.0 / lam) + K * (np.log(lam) / lam)

    def tangent(self, strain: float) -> float:
        lam = self._stretch(strain)
        mu = self.mu
        K = self.K
        lam2 = lam * lam
        return mu * (1.0 + 1.0 / lam2) + K * (1.0 - np.log(lam)) / lam2

def make_material(material_spec: dict[str, Any]) -> Material1D:

    mtype = material_spec.get("type", "linear_elastic").lower()
    params = material_spec.get("parameters", {})
    rho = material_spec.get("density", None)

    if mtype in {"linear", "ELASTIC", "elastic", "linear_elastic", "linear-elastic"}:
        return LinearElastic1D(
            E=params["E"],
            alpha=params.get("alpha", 0.0),
            dT=params.get("dT", 0.0),
            rho=rho,
        )
    
    elif mtype in {"neohookean", "neo-hookean", "nh"}:

        if "mu" in params:
            mu = float(params["mu"])
            K = float(params.get("K", 0.0)) 
        elif "E" in params and "nu" in params:
            E = float(params["E"])
            nu = float(params["nu"])
            mu = E / (2.0 * (1.0 + nu))
            K = float(params.get("K", 0.0))  
        else:
            raise ValueError(
                "NeoHookean1D requires either ('mu', ['K']) or ('E','nu', ['K'])."
            )
        return NeoHookean1D(mu=mu, K=K, rho=rho)
    
    raise NotImplementedError(f"Material type {mtype!r} not supported.")


_GAUSS_XI_2 = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
_GAUSS_W_2  = np.array([1.0, 1.0])

def _shape_lin(xi: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Linear 2-node bar on parent [-1,1]."""
    N = np.array([0.5 * (1 - xi), 0.5 * (1 + xi)], dtype=float)
    dN_dxi = np.array([-0.5, 0.5], dtype=float)
    return N, dN_dxi

def _kinematics_1d(xe: NDArray[float], ue: Optional[NDArray[float]], xi: float
                   ) -> tuple[float, float, NDArray[float], float]:
    
    N, dN_dxi = _shape_lin(xi)
    h = float(xe[1] - xe[0])
    if np.isclose(h, 0.0):
        raise ValueError("Zero-length element.")
    J = abs(h) / 2.0
    dN_dx = dN_dxi / J
    B = dN_dx  
    x = float(N @ xe)
    strain = float(B @ ue) if ue is not None else 0.0
    return x, J, B, strain

def _as_area_func(A: float | Callable[[float], float]) -> Callable[[float], float]:
    if callable(A):
        return A
    a = float(A)
    return lambda x: a

def element_stiffness_bar1d(
    xe: NDArray[float],                  
    A: float | Callable[[float], float],     
    material: Material1D,
    ue: Optional[NDArray[float]] = None,
    n_gauss: int = 2,
) -> NDArray[float]:
    """Ke = ∑ B^T C B A(x) J w"""
    if n_gauss != 2:
        raise NotImplementedError("Only 2-pt Gauss implemented for now.")
    Ke = np.zeros((2, 2), dtype=float)
    A_of_x = _as_area_func(A)

    if ue is None:
        ue = np.zeros(2, dtype=float)

    for xi, w in zip(_GAUSS_XI_2, _GAUSS_W_2):
        x, J, B, strain = _kinematics_1d(xe, ue, xi)
        C = material.tangent(strain)
        Ke += np.outer(B, B) * C * A_of_x(x) * J * w
    return Ke

def element_internal_force_bar1d(
    xe: NDArray[float],                     
    ue: NDArray[float],                
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
    xe: NDArray[float],                       
    q_of_x: float | Callable[[float], float],   
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
        J = abs(h) / 2.0
        x = float(N @ xe)
        fext += N * qx(x) * J * w
    return fext

def make_q_constant(q: float | int) -> Callable[[float], float]:
    """Return a spatially constant distributed load q(x) ≡ q."""
    q_const = float(q)

    def q_of_x(x: float) -> float:
        return q_const

    return q_of_x


def make_q_from_table(xq_pairs: Any) -> Callable[[float], float]:

    xq = np.asarray(xq_pairs, dtype=float)
    if xq.ndim != 2 or xq.shape[1] != 2:
        raise ValueError("xq_pairs must be an array-like of shape (n, 2)")

    xs = xq[:, 0]
    qs = xq[:, 1]

    def q_of_x(x: float) -> float:
        return float(np.interp(float(x), xs, qs))

    return q_of_x


def make_q_from_equation(expr: str, L: float) -> Callable[[float], float]:

    L_float = float(L)
    allowed_globals = {"np": np, "pi": float(np.pi), "L": L_float}

    def q_of_x(x: float) -> float:
        local_env = {"x": float(x)}
        return float(eval(expr, {"__builtins__": {}}, {**allowed_globals, **local_env}))

    return q_of_x
