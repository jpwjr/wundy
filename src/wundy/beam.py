import numpy as np
from numpy.typing import NDArray
from typing import Any, List, Dict

from .schemas import DIRICHLET, NEUMANN
from .first import global_dof 


def element_stiffness_euler_bernoulli(E: float, I: float, L: float) -> NDArray[np.float64]:
    """
    2-node Euler–Bernoulli beam element, Hermite (w, θ) DOFs:

        dofs per element: [w1, θ1, w2, θ2]

    Standard closed-form stiffness (local coordinates):

        Ke = (E I / L^3) * [[  12,   6L,  -12,   6L],
                            [  6L, 4L²,  -6L, 2L²],
                            [ -12,  -6L,  12,  -6L],
                            [  6L, 2L²,  -6L, 4L²]]
    """
    L = float(L)
    if L <= 0.0:
        raise ValueError(f"Beam element length must be positive, got L={L}")

    EI_over_L3 = (E * I) / (L ** 3)
    L2 = L * L

    Ke = EI_over_L3 * np.array(
        [
            [12.0,   6.0 * L,  -12.0,   6.0 * L],
            [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2],
            [-12.0,  -6.0 * L,  12.0,  -6.0 * L],
            [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2],
        ],
        dtype=float,
    )
    return Ke


def element_consistent_uniform_load(q: float, L: float) -> NDArray[np.float64]:
    """
    Consistent nodal load vector for a uniform transverse load q (force/length)
    over a 2-node Hermite beam element of length L:

        fe = q L / 12 * [6, L, 6, -L]

    Sign convention: positive q produces positive deflection w.
    """
    L = float(L)
    q = float(q)
    factor = q * L / 12.0
    fe = factor * np.array([6.0, L, 6.0, -L], dtype=float)
    return fe


def assemble_beam_chain(
    coords: NDArray[np.float64],      
    connect: NDArray[np.int_],      
    E: float,
    I: float,
    q: float | None = None,           
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    dof_per_node = 2
    num_node = coords.shape[0]
    num_dof = num_node * dof_per_node

    K = np.zeros((num_dof, num_dof), dtype=float)
    F = np.zeros(num_dof, dtype=float)

    for e in range(connect.shape[0]):
        nodes = connect[e]  # [n1, n2]
        n1, n2 = int(nodes[0]), int(nodes[1])

        x1 = float(coords[n1, 0])
        x2 = float(coords[n2, 0])
        L = x2 - x1
        if L <= 0.0:
            raise ValueError(f"Beam element [{n1}, {n2}] has non-positive length L={L}")

        Ke = element_stiffness_euler_bernoulli(E, I, L)

        if q is not None:
            fe = element_consistent_uniform_load(q, L)
        else:
            fe = np.zeros(4, dtype=float)

        eft = [
            global_dof(n1, 0, dof_per_node),  # w1
            global_dof(n1, 1, dof_per_node),  # θ1
            global_dof(n2, 0, dof_per_node),  # w2
            global_dof(n2, 1, dof_per_node),  # θ2
        ]

        K[np.ix_(eft, eft)] += Ke
        F[eft] += fe

    return K, F


def first_beam_code(
    coords: NDArray[np.float64],       
    connect: NDArray[np.int_],       
    E: float,
    I: float,
    q: float | None,
    bcs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Simple Euler–Bernoulli beam solver, parallel in spirit to first_fe_code
    but with 2 DOFs per node: w (transverse disp) and θ (rotation). :contentReference[oaicite:2]{index=2}

    Inputs:
      - coords : nodal coordinates (N, 1)
      - connect: element connectivity (n_elem, 2) with 0-based node indices
      - E, I   : beam material stiffness and second moment of area
      - q      : uniform transverse load on all elements (force/length), or None
      - bcs    : list of BC dicts, each of the form:
            {"type": DIRICHLET/NEUMANN,
             "node": int (0-based),
             "local_dof": 0 for w, 1 for θ,
             "value": float}

    Returns:
      {"dofs": u, "stiff": K, "force": F}
    """
    dof_per_node = 2
    num_node = coords.shape[0]
    num_dof = num_node * dof_per_node

    K, F = assemble_beam_chain(coords, connect, E, I, q)

    for bc in bcs:
        if bc["type"] == NEUMANN:
            node = int(bc["node"])
            local = int(bc["local_dof"])
            I = global_dof(node, local, dof_per_node)
            F[I] += float(bc["value"])

    prescribed_dofs: list[int] = []
    prescribed_vals: list[float] = []
    for bc in bcs:
        if bc["type"] == DIRICHLET:
            node = int(bc["node"])
            local = int(bc["local_dof"])
            I = global_dof(node, local, dof_per_node)
            prescribed_dofs.append(I)
            prescribed_vals.append(float(bc["value"]))

    prescribed_dofs_arr = np.asarray(prescribed_dofs, dtype=int)
    prescribed_vals_arr = np.asarray(prescribed_vals, dtype=float)

    all_dofs = np.arange(num_dof, dtype=int)
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs_arr)

    u = np.zeros(num_dof, dtype=float)

    if free_dofs.size > 0:
        Kff = K[np.ix_(free_dofs, free_dofs)]
        Kfp = K[np.ix_(free_dofs, prescribed_dofs_arr)]
        Ff = F[free_dofs] - Kfp @ prescribed_vals_arr
        uf = np.linalg.solve(Kff, Ff)

        u[free_dofs] = uf
        u[prescribed_dofs_arr] = prescribed_vals_arr
    else:
        u[prescribed_dofs_arr] = prescribed_vals_arr

    return {"dofs": u, "stiff": K, "force": F}
