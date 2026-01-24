"""
first.py — 1D bar finite element assembler/solver (and tests)

This module provides a minimal “first FE code” for a 1D axial bar mesh with
two‑node linear elements. It assembles the global stiffness matrix K and load
vector F from:
- nodal coordinates `coords` (shape: [N, 1]),
- element blocks that include connectivity, material, and section properties,
- boundary conditions (Dirichlet and Neumann),
- optional distributed loads (body force along the bar).

It then applies boundary conditions via symmetry‑preserving elimination and
solves for the nodal displacements.

Below the implementation you’ll find a compact pytest suite that verifies:
- correct assembly for a uniform 1D chain with a point load,
- stiffness symmetry/shape and positive definiteness for a constrained case,
- distributed load assembly for a simple body force (“BX”) case,
- robust error handling for zero‑length elements and bad distributed‑load input,
- the `global_dof` mapping behavior.

The tests are written to be self‑contained (no YAML/UI layer)—they construct
the dictionaries/lists that `first_fe_code` expects directly.

Final Project Assignment:
 1) Euler-Beam with Test
 2)* Modify Code to Accept "Arbitrary" Distributed Load and provide method of manufactured Soln. Verification
    **Bonus if the MMS is for the beam element.
    Must Use Neo-Hookean Material Model.

 *most important 
 **Optional

 "Arbitrary" Distributed Load is a function of x
    could be a table of x vs q or an equation
Dload:
    name:...
    value: [[x0, q0], [x1, q1], ...]  Or q(x) = "sin(pi*x/L)"
    input_type: "table" Or "equation"

In Code Changes:
-Schema needs to recognize new field: input_type [default: scalar]
-Preprocessor needs to handle the new input_type
-element force update needs to handle non constant distributed load
    -if table: interpolate to get q at gauss points
    -if equation: evaluate at gauss points
        element_force(xe, q):
            for xi,w in gauss_info(num_gauss):
                N=shape_function(xi,xe)
                x = N*xe
                J=jacobian(  )
                qe= q(x)  # if table -> interpolate, if equation -> eval
                fe=fe+qe*N*w*J

            input_type:
                table: np.interpld(x,q)
                equation: something=eval(q,{x:xp}) #what is x -> it is a location in your mesh- a scalar identifier
-Possible tests:
    -does it interpolate correctly
    dload:
        name: constant
        type: table
        value: [[0,12]]

Generalized BC's notes from class:
we now have Kij_bc * uj = Fi_bc
for generalized we can have
Kij_bc = Kij, Fi_bc = Fi
Except on gamma_u, gamma_uprime
    where Kuu_bc = Kuu - alpha_u/beta_u, Fu_bc = Fu + gamma_u/beta_u

"""

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .schemas import DIRICHLET, NEUMANN
from .elematl import (
    make_material,
    element_stiffness_bar1d,
    element_external_body_bar1d,
    element_internal_force_bar1d,
)

def _make_area_of_x(
    props: dict[str, Any],
    x_min: float,
    x_max: float,
) -> Callable[[float], float]:
    """
    Returns A(x) based on whatever is provided in element properties.

    Supported (in order of precedence):
      1) props["area_func"]: callable(x)->area
      2) props["area0"] and props["area1"]: linear taper over [x_min, x_max]
      3) props["area_start"] and props["area_end"]: linear taper over [x_min, x_max]
      4) props["area"]: constant
    """
    if "area_func" in props and callable(props["area_func"]):
        area_func = props["area_func"]

        def A_of_x(x: float, f=area_func) -> float:
            return float(f(x))

        return A_of_x

    # Linear taper definitions (global over the bar domain)
    if ("area0" in props and "area1" in props) or ("area_start" in props and "area_end" in props):
        a0 = float(props.get("area0", props.get("area_start")))
        a1 = float(props.get("area1", props.get("area_end")))
        L = float(x_max - x_min) if x_max != x_min else 1.0

        def A_of_x(x: float, a0=a0, a1=a1, x0=x_min, L=L) -> float:
            xi = (float(x) - x0) / L
            return a0 + (a1 - a0) * xi

        return A_of_x

    # Fallback constant
    A_const = float(props["area"])

    def A_of_x(x: float, A_const=A_const) -> float:
        return A_const

    return A_of_x


def _element_area_midpoint(
    xe: NDArray[np.float64],
    props: dict[str, Any],
    x_min: float,
    x_max: float,
) -> float:
    """
    Element area computed from the midpoint coordinate of the element.
    xe is the element nodal coordinate vector (shape (2,) for 2-node bar).
    """
    x_mid = 0.5 * (float(xe[0]) + float(xe[-1]))
    A_of_x = _make_area_of_x(props, x_min, x_max)
    return float(A_of_x(x_mid))


# -----------------------------
# Assembly
# -----------------------------
def assemble_distributed_loads(
    coords: NDArray[np.float64],
    blocks: list[dict],
    dload: list[dict] | None,
    materials: dict[str, Any],
    block_elem_map: dict[int, tuple[int, int]],
    n_gauss: int = 2,
) -> NDArray[np.float64]:
    num_node = coords.shape[0]
    dof_per_node = 1
    num_dof = num_node * dof_per_node

    F = np.zeros(num_dof, dtype=float)

    # Global domain (used for linear taper definitions)
    x_min = float(np.min(coords[:, 0]))
    x_max = float(np.max(coords[:, 0]))

    for dl in dload or []:
        dtype = dl["type"].upper()

        if dtype == "BX":
            direction = np.asarray(dl["direction"], dtype=float)
            sign = float(direction[0])

            input_type = dl.get("input_type", "SCALAR").upper()
            if input_type == "SCALAR":
                q0 = float(dl["value"]) * sign

                def q_of_x(x, q0=q0) -> float:
                    return q0
            else:
                q_func = dl["q_func"]

                def q_of_x(x, q_func=q_func, sign=sign) -> float:
                    return float(sign * q_func(x))

            for e in dl["elements"]:
                ib, ie = block_elem_map[e]
                block = blocks[ib]
                nodes = np.asarray(block["connect"][ie], dtype=int)
                xe = coords[nodes, 0]

                # Uses Gauss quadrature inside element_external_body_bar1d when n_gauss is provided
                fe = element_external_body_bar1d(xe, q_of_x, n_gauss=n_gauss)

                for a, node in enumerate(nodes):
                    F[node] += fe[a]

        elif dtype == "GRAV":
            direction = np.asarray(dl["direction"], dtype=float)
            sign = float(direction[0])

            for e in dl["elements"]:
                ib, ie = block_elem_map[e]
                block = blocks[ib]

                mat_spec = materials[block["material"]]
                rho = float(mat_spec["density"])

                nodes = np.asarray(block["connect"][ie], dtype=int)
                xe = coords[nodes, 0]

                # Midpoint area for this element (variable A)
                props = block["element"]["properties"]
                A_mid = _element_area_midpoint(xe, props, x_min=x_min, x_max=x_max)

                g = 9.81
                q0 = rho * A_mid * g * sign

                def q_of_x(x, q0=q0) -> float:
                    return q0

                # Uses Gauss quadrature inside element_external_body_bar1d when n_gauss is provided
                fe = element_external_body_bar1d(xe, q_of_x, n_gauss=n_gauss)

                for a, node in enumerate(nodes):
                    F[node] += fe[a]

    return F


def first_fe_code(
    coords: NDArray[float],
    blocks: list[dict],
    bcs: list[dict],
    dload: list[dict] | None,
    materials: dict[str, Any],
    block_elem_map: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    dof_per_node = 1
    num_node = coords.shape[0]
    num_dof = num_node * dof_per_node

    K = np.zeros((num_dof, num_dof), dtype=float)
    F = assemble_distributed_loads(
        coords=coords,
        blocks=blocks,
        dload=dload,
        materials=materials,
        block_elem_map=block_elem_map,
        n_gauss=2,
    )

    # Global domain (used for linear taper definitions)
    x_min = float(np.min(coords[:, 0]))
    x_max = float(np.max(coords[:, 0]))

    for bc in bcs:
        if bc["type"] == NEUMANN:
            for node in bc["nodes"]:
                F[node] += bc["value"]

    for block in blocks:
        material_obj = make_material(materials[block["material"]])
        props = block["element"]["properties"]

        for nodes in block["connect"]:
            nodes = np.asarray(nodes, dtype=int)
            eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            xe_vec = coords[nodes, 0]

            # Midpoint area for this element (variable A)
            A_mid = _element_area_midpoint(xe_vec, props, x_min=x_min, x_max=x_max)

            # Gauss quadrature is used inside element_stiffness_bar1d via n_gauss
            ke = element_stiffness_bar1d(xe_vec, A_mid, material_obj, ue=None, n_gauss=2)
            K[np.ix_(eft, eft)] += ke

    prescribed_dofs: list[int] = []
    prescribed_vals: list[float] = []
    for bc in bcs:
        if bc["type"] == DIRICHLET:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                prescribed_dofs.append(I)
                prescribed_vals.append(bc["value"])

    all_dofs = np.arange(num_dof)
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    if free_dofs.size == 0:
        dofs = np.zeros(num_dof, dtype=float)
        dofs[prescribed_dofs] = prescribed_vals
        return {"dofs": dofs, "stiff": K, "force": F}

    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfp = K[np.ix_(free_dofs, prescribed_dofs)]
    Ff = F[free_dofs] - np.dot(Kfp, prescribed_vals)
    uf = np.linalg.solve(Kff, Ff)

    dofs = np.zeros(num_dof, dtype=float)
    dofs[free_dofs] = uf
    dofs[prescribed_dofs] = prescribed_vals

    return {"dofs": dofs, "stiff": K, "force": F}


def newton_solve_bar1d(
    coords: NDArray[float],
    blocks: list[dict],
    bcs: list[dict],
    dload: list[dict] | None,
    materials: dict[str, Any],
    block_elem_map: dict[int, tuple[int, int]],
    tol: float = 1e-10,
    max_iter: int = 25,
) -> dict[str, Any]:
    dof_per_node = 1
    num_node = coords.shape[0]
    num_dof = num_node * dof_per_node

    u = np.zeros(num_dof, dtype=float)

    prescribed_dofs: list[int] = []
    prescribed_vals: list[float] = []
    for bc in bcs:
        if bc["type"] == DIRICHLET:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                prescribed_dofs.append(I)
                prescribed_vals.append(float(bc["value"]))

    prescribed_dofs = np.asarray(prescribed_dofs, dtype=int)
    prescribed_vals = np.asarray(prescribed_vals, dtype=float)

    all_dofs = np.arange(num_dof, dtype=int)
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    if prescribed_dofs.size > 0:
        u[prescribed_dofs] = prescribed_vals

    F_ext = assemble_distributed_loads(
        coords=coords,
        blocks=blocks,
        dload=dload,
        materials=materials,
        block_elem_map=block_elem_map,
        n_gauss=2,
    )

    for bc in bcs:
        if bc["type"] == NEUMANN:
            for node in bc["nodes"]:
                F_ext[node] += bc["value"]

    # Global domain (used for linear taper definitions)
    x_min = float(np.min(coords[:, 0]))
    x_max = float(np.max(coords[:, 0]))

    for _it in range(max_iter):
        K = np.zeros((num_dof, num_dof), dtype=float)
        R_int = np.zeros(num_dof, dtype=float)

        for block in blocks:
            material_obj = make_material(materials[block["material"]])
            props = block["element"]["properties"]

            for nodes in block["connect"]:
                nodes = np.asarray(nodes, dtype=int)
                eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
                xe_vec = coords[nodes, 0]
                ue = u[eft]

                # Midpoint area for this element (variable A)
                A_mid = _element_area_midpoint(xe_vec, props, x_min=x_min, x_max=x_max)

                # Gauss quadrature is used inside these element routines via n_gauss
                ke = element_stiffness_bar1d(xe_vec, A_mid, material_obj, ue=ue, n_gauss=2)
                fint_e = element_internal_force_bar1d(xe_vec, ue, A_mid, material_obj, n_gauss=2)

                K[np.ix_(eft, eft)] += ke
                R_int[eft] += fint_e

        if prescribed_dofs.size > 0:
            u[prescribed_dofs] = prescribed_vals

        residual = F_ext - R_int
        res_f = residual[free_dofs]
        K_ff = K[np.ix_(free_dofs, free_dofs)]

        du_f = np.linalg.solve(K_ff, res_f)
        u[free_dofs] += du_f

        if np.linalg.norm(du_f, ord=2) < tol:
            break
    else:
        du_norm = np.linalg.norm(du_f, ord=2)
        raise RuntimeError(
            f"Newton solver did not converge in {max_iter} iterations, the last du was {du_norm}."
        )

    return {"dofs": u, "stiff": K, "force": F_ext}


def global_dof(node: int, local_dof: int, dof_per_node: int) -> int:
    return node * dof_per_node + local_dof
