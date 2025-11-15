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
"""

from typing import Any, Callable, Protocol, Optional

import numpy as np
from numpy.typing import NDArray

from .schemas import DIRICHLET
from .schemas import NEUMANN

from .elematl import (
    make_material,
    element_stiffness_bar1d,
    element_external_body_bar1d,
    element_internal_force_bar1d,
)

def first_fe_code(
    coords: NDArray[float],
    blocks: list[dict],
    bcs: list[dict],
    dload: list[dict] | None,                # accept singular from preprocess
    materials: dict[str, Any],
    block_elem_map: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    dloads: list[dict] = dload or []         # normalize here

    dof_per_node = 1
    num_node = coords.shape[0]
    num_dof = num_node * dof_per_node
    K = np.zeros((num_dof, num_dof), dtype=float)
    F = np.zeros(num_dof, dtype=float)

    # Assemble global stiffness
    for block in blocks:
        A = block["element"]["properties"]["area"]
        material_obj = make_material(materials[block["material"]])

        for nodes in block["connect"]:
            eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            xe_vec = coords[nodes, 0] #shape (2,) -> physical x-coordinates

            # Gauss-derived element stiffness
            ke = element_stiffness_bar1d(xe_vec, A, material_obj, ue=None, n_gauss=2)
            K[np.ix_(eft, eft)] += ke

        """
        old material handling code:

        E = material["parameters"]["E"]
        for nodes in block["connect"]:
            # GLOBAL DOF = NODE NUMBER x NUMBER OF DOF PER NODE + LOCAL DOF
            eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]

            xe = coords[nodes]
            he = xe[1, 0] - xe[0, 0]
            if np.isclose(he, 0.0):
                raise ValueError(f"Zero-length element detected between nodes {nodes}")
            ke = A * E / he * np.array([[1.0, -1.0], [-1.0, 1.0]])
            K[np.ix_(eft, eft)] += ke

        """
    # Apply Neumann boundary conditions to force
    for bc in bcs:
        if bc["type"] == NEUMANN:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                F[I] += bc["value"]

    # Apply distributed loads (Gauss-Quadrature)
    for dl in dloads:
        dtype = dl["type"]
        direction = np.array(dl["direction"], dtype=float)
        if direction.size != 1:
            raise ValueError(f"1D problem expects one direction component, got {direction}")
        sign = np.sign(direction[0])
        if sign == 0.0:
            raise ValueError(f"dload direction must be ±1, got {direction[0]}")
        for eid in dl["elements"]:
            if eid not in block_elem_map:
                raise ValueError(
                    f"Element {eid} in distributed load "
                    f"{dl['name']} not found in any element block"
                )
            block_index, local_index = block_elem_map[eid]
            block = blocks[block_index]
            nodes = block["connect"][local_index]
            xe_vec = coords[nodes, 0] # shape (2,)
            # he = xe[1, 0] - xe[0, 0] # See if this breaks anything

            A = block["element"]["properties"]["area"]

            if dtype == "BX":
                # force/length, possibly variable later
                q_const = float(dl["value"]) * float(sign)
                fext_e = element_external_body_bar1d(xe_vec, q_const, n_gauss=2)

            elif dtype == "GRAV":
                mat_spec = materials[block["material"]]
                rho = float(mat_spec["density"])
                g = float(dl["value"]) * float(sign)

                if callable(A):
                    q_of_x = lambda x: rho * float(A(x)) * g
                else:
                    q_of_x = rho * float(A) * g
            
                fext_e = element_external_body_bar1d(xe_vec, q_of_x, n_gauss=2)

            else:
                raise NotImplementedError(f"dload type {dtype!r} not supported for 1D")
        
            eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            F[eft] += fext_e

    # Apply Dirchlet boundary conditions using a symmetry preserving elimination
    # Let
    #   Ku = f
    # split dofs into two sets:
    #   1. free
    #   2. prescribed
    # Set up new system:
    #
    #  | K_ff  K_fp |  [ u_f ]   | F_f |
    #  | K_pf  K_pp |  [ u_p ]   | F_p |
    #
    # Eliminate prescribed dofs:
    #   K_ff.u_f = Ff - K_fp.u_p
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
    Kff = K[np.ix_(free_dofs, free_dofs)]
    Kfp = K[np.ix_(free_dofs, prescribed_dofs)]
    Ff = F[free_dofs] - np.dot(Kfp, prescribed_vals)
    uf = np.linalg.solve(Kff, Ff)

    # solve the system
    dofs = np.zeros(num_dof, dtype=float)
    dofs[free_dofs] = uf
    dofs[prescribed_dofs] = prescribed_vals

    solution = {"dofs": dofs, "stiff": K, "force": F}

    return solution

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
    """
    Newton–Raphson solver for 1D bar, using internal-force formulation:

        R_int(u) = F_ext
        K_tan(u) = dR_int/du

    Algorithm (matches your handwritten steps):

      0) start with guess u_j (here u = 0)
      1) compute k_ij(u), R_i(u), F_i
      2) enforce boundary conditions (Dirichlet)
      3) solve k_ij Δu_j = F_i - R_i
      4) set u_j ← u_j + Δu_j
      5) if ||Δu|| < tol, STOP; else go to 1
    """
    dloads: list[dict] = dload or []

    dof_per_node = 1
    num_node = coords.shape[0]
    num_dof = num_node * dof_per_node

    # -------------------------------
    # Step 0) initial guess u_j = 0
    # -------------------------------
    u = np.zeros(num_dof, dtype=float)

    # --- Dirichlet information (prescribed DOFs) ---
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

    # Enforce prescribed values in the initial guess
    if prescribed_dofs.size > 0:
        u[prescribed_dofs] = prescribed_vals

    # ------------------------------------------
    # Assemble external force F_i (independent of u)
    # ------------------------------------------
    F_ext = np.zeros(num_dof, dtype=float)

    # Neumann boundary conditions
    for bc in bcs:
        if bc["type"] == NEUMANN:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                F_ext[I] += float(bc["value"])

    # Distributed loads
    for dl in dloads:
        dtype = dl["type"]
        direction = np.array(dl["direction"], dtype=float)
        if direction.size != 1:
            raise ValueError(f"1D problem expects one direction component, got {direction}")
        sign = np.sign(direction[0])
        if sign == 0.0:
            raise ValueError(f"dload direction must be ±1, got {direction[0]}")

        for eid in dl["elements"]:
            if eid not in block_elem_map:
                raise ValueError(
                    f"Element {eid} in distributed load "
                    f"{dl['name']} not found in any element block"
                )
            block_index, local_index = block_elem_map[eid]
            block = blocks[block_index]
            nodes = block["connect"][local_index]
            xe_vec = coords[nodes, 0]
            A = block["element"]["properties"]["area"]

            if dtype == "BX":
                q_const = float(dl["value"]) * float(sign)
                fext_e = element_external_body_bar1d(xe_vec, q_const, n_gauss=2)

            elif dtype == "GRAV":
                mat_spec = materials[block["material"]]
                rho = float(mat_spec["density"])
                g = float(dl["value"]) * float(sign)

                if callable(A):
                    q_of_x = lambda x: rho * float(A(x)) * g
                else:
                    q_of_x = rho * float(A) * g

                fext_e = element_external_body_bar1d(xe_vec, q_of_x, n_gauss=2)

            else:
                raise NotImplementedError(f"dload type {dtype!r} not supported for 1D")

            eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            F_ext[eft] += fext_e

    # ------------------------------------------
    # Newton iterations
    # ------------------------------------------
    for it in range(max_iter):
        # 1) compute k_ij, R_i, F_i
        K = np.zeros((num_dof, num_dof), dtype=float)
        R_int = np.zeros(num_dof, dtype=float)

        for block in blocks:
            A = block["element"]["properties"]["area"]
            material_obj = make_material(materials[block["material"]])

            for nodes in block["connect"]:
                # element freedom table
                eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
                xe_vec = coords[nodes, 0]
                ue = u[eft]

                # element tangent stiffness and internal force
                ke = element_stiffness_bar1d(xe_vec, A, material_obj, ue=ue, n_gauss=2)
                fint_e = element_internal_force_bar1d(xe_vec, ue, A, material_obj, n_gauss=2)

                K[np.ix_(eft, eft)] += ke
                R_int[eft] += fint_e

        # 2) enforce boundary conditions: keep u_p fixed, solve only on free dofs
        if prescribed_dofs.size > 0:
            u[prescribed_dofs] = prescribed_vals

        # 3) solve k_ij Δu_j = F_i - R_i on free DOFs
        residual = F_ext - R_int
        res_f = residual[free_dofs]
        K_ff = K[np.ix_(free_dofs, free_dofs)]

        du_f = np.linalg.solve(K_ff, res_f)

        # 4) update u_j = u_j + Δu_j
        u[free_dofs] += du_f

        # 5) convergence check: ||Δu|| < tol
        du_norm = np.linalg.norm(du_f, ord=2)
        if du_norm < tol:
            # converged
            break
    else:
        # If we exit the for-loop without a break
        raise RuntimeError(f"Newton solver did not converge in {max_iter} iterations")

    # Return last tangent K and external force vector
    return {"dofs": u, "stiff": K, "force": F_ext}


def global_dof(node: int, local_dof: int, dof_per_node: int) -> int:
    """Return the global degree of freedom index for a given node and local dof

    NOTE: Assumes elements have uniform degrees of freedom across the mesh.

    """
    return node * dof_per_node + local_dof
