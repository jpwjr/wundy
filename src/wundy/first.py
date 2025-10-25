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

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .schemas import DIRICHLET
from .schemas import NEUMANN


def first_fe_code(
    coords: NDArray[float],
    blocks: list[dict],
    bcs: list[dict],
    dloads: list[dict],
    materials: dict[str, Any],
    block_elem_map: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    """Assemble and solve a 1D bar finite‑element system.

    Parameters
    ----------
    coords
        Array of nodal coordinates with shape (N, 1). Node indices are expected
        to be 0‑based and used directly for indexing (e.g., connectivity [i, j]
        means coords[i] and coords[j]).
    blocks
        A list of element‑block dicts. Each block must contain:
          - 'material': key into `materials`
          - 'connect': list of 2‑node element connectivities (node indices)
          - 'element': {'properties': {'area': float}, ...} for 1D bar
    bcs
        Boundary conditions as a list of dicts. Supported types are the constants
        imported from .schemas: DIRICHLET and NEUMANN. Each bc requires:
          - 'type': DIRICHLET or NEUMANN
          - 'nodes': list[int] of node indices
          - 'local_dof': int (0 for 1D bars)
          - 'value': float (prescribed displacement for DIRICHLET; force for NEUMANN)
    dloads
        Distributed loads list. Each item requires:
          - 'type': 'BX' (body force along x) or 'GRAV' (uses material density)
          - 'direction': length‑1 list/array with ±1 indicating sign
          - 'elements': list[int] of element IDs
          - 'value': float magnitude
        'GRAV' additionally expects materials[block['material']]['density'].
    materials
        Mapping of material names to dicts that include at least
        {'parameters': {'E': Youngs modulus}}, and optionally 'density' for GRAV.
    block_elem_map
        Mapping from global element id (int) to a tuple (block_index, local_index)
        so we can look up the owning block and its connectivity row.

    Returns
    -------
    dict
        A dict with:
          - 'dofs': (N,) ndarray of nodal displacements
          - 'stiff': (N,N) ndarray global stiffness matrix
          - 'force': (N,) ndarray global force vector

    Raises
    ------
    ValueError
        On zero‑length elements or malformed distributed load directions.
    NotImplementedError
        If an unsupported distributed‑load type is requested.

    Notes
    -----
    The implementation assumes 1 DOF per node (axial displacement), uniform over
    the mesh. Dirichlet elimination preserves symmetry of the reduced system.
    """
    dof_per_node = 1
    num_node = coords.shape[0]
    num_dof = num_node * dof_per_node
    K = np.zeros((num_dof, num_dof), dtype=float)
    F = np.zeros(num_dof, dtype=float)

    # Assemble global stiffness
    for block in blocks:
        A = block["element"]["properties"]["area"]
        material = materials[block["material"]]
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

    # Apply Neumann boundary conditions to force
    for bc in bcs:
        if bc["type"] == NEUMANN:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                F[I] += bc["value"]

    # Apply distributed loads
    for dload in dloads:
        dtype = dload["type"]
        direction = np.array(dload["direction"], dtype=float)
        if direction.size != 1:
            raise ValueError(f"1D problem expects one direction component, got {direction}")
        sign = np.sign(direction[0])
        if sign == 0.0:
            raise ValueError(f"dload direction must be ±1, got {direction[0]}")
        for eid in dload["elements"]:
            if eid not in block_elem_map:
                raise ValueError(
                    f"Element {eid} in distributed load "
                    f"{dload['name']} not found in any element block"
                )
            block_index, local_index = block_elem_map[eid]
            block = blocks[block_index]
            nodes = block["connect"][local_index]
            xe = coords[nodes]
            he = xe[1, 0] - xe[0, 0]
            A = block["element"]["properties"]["area"]
            if dtype == "BX":
                q = dload["value"] * sign
            elif dtype == "GRAV":
                mat = materials[block["material"]]
                rho = mat["density"]
                q = rho * A * dload["value"] * sign
            else:
                raise NotImplementedError(f"dload type {dtype!r} not supported for 1D")
            eft = [global_dof(n, j, dof_per_node) for n in nodes for j in range(dof_per_node)]
            qe = q * he / 2 * np.ones(2)
            F[eft] += qe

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


def global_dof(node: int, local_dof: int, dof_per_node: int) -> int:
    """Compute the global DOF index for a given node and local DOF.

    For a mesh with a uniform number of DOFs per node, the mapping is:
        global = node * dof_per_node + local_dof

    Parameters
    ----------
    node
        Zero‑based node index.
    local_dof
        Local DOF index at the node (0 for a 1D bar).
    dof_per_node
        Number of DOFs per node (assumed uniform).

    Returns
    -------
    int
        The global DOF index.
    """
    return node * dof_per_node + local_dof


# ---------------------------
# Pytest unit tests (inline)
# ---------------------------

def _uniform_chain_input(E: float = 10.0, A: float = 1.0):
    """Create a simple 5‑node, 4‑element uniform chain input for tests.

    Nodes at x = [0, 1, 2, 3, 4], elements [0-1, 1-2, 2-3, 3-4], single DOF per node.
    Returns (coords, blocks, bcs, dloads, materials, block_elem_map).
    """
    coords = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=float)
    connect = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=int)
    blocks = [{
        "material": "mat-1",
        "connect": connect,
        "element": {"properties": {"area": A}},
    }]
    materials = {"mat-1": {"parameters": {"E": E}}}
    # Map global element IDs 0..3 to (block_index=0, local_index)
    block_elem_map = {eid: (0, eid) for eid in range(connect.shape[0])}
    # No distributed loads by default; set in specific tests
    dloads = []
    # Boundary conditions: fix node 0 in x (local_dof=0), and apply a point load at node 4
    bcs = [
        {"type": DIRICHLET, "nodes": [0], "local_dof": 0, "value": 0.0},
        {"type": NEUMANN,   "nodes": [4], "local_dof": 0, "value": 2.0},
    ]
    return coords, blocks, bcs, dloads, materials, block_elem_map


def test_uniform_chain_solution_matches_reference():
    coords, blocks, bcs, dloads, materials, block_elem_map = _uniform_chain_input()
    soln = first_fe_code(coords, blocks, bcs, dloads, materials, block_elem_map)

    dofs = soln["dofs"]
    K = soln["stiff"]
    F = soln["force"]

    np.testing.assert_allclose(dofs, [0.0, 0.2, 0.4, 0.6, 0.8], atol=1e-12)
    np.testing.assert_allclose(F, [0.0, 0.0, 0.0, 0.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(
        K,
        [[10, -10,   0,   0,   0],
         [-10, 20, -10,   0,   0],
         [  0,-10,  20, -10,   0],
         [  0,  0, -10,  20, -10],
         [  0,  0,   0, -10,  10]],
        atol=1e-12,
    )


def test_stiffness_symmetry_and_shapes():
    coords, blocks, bcs, dloads, materials, block_elem_map = _uniform_chain_input()
    soln = first_fe_code(coords, blocks, bcs, dloads, materials, block_elem_map)

    dofs = soln["dofs"]
    K = soln["stiff"]
    F = soln["force"]
    n = dofs.size

    assert K.shape == (n, n)
    assert F.shape == (n,)
    np.testing.assert_allclose(K, K.T, atol=1e-12)


def test_positive_definite_with_constraints():
    coords, blocks, bcs, dloads, materials, block_elem_map = _uniform_chain_input()
    soln = first_fe_code(coords, blocks, bcs, dloads, materials, block_elem_map)
    K = soln["stiff"]
    # For this constrained, connected chain K should be SPD
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals > 0.0)


def test_distributed_load_BX_assembly():
    coords, blocks, bcs, dloads, materials, block_elem_map = _uniform_chain_input()
    # Remove the point load; keep only Dirichlet, add uniform BX with q=1 in +x
    bcs = [{"type": DIRICHLET, "nodes": [0], "local_dof": 0, "value": 0.0}]
    dloads = [{
        "name": "q+1",
        "type": "BX",
        "direction": [1],     # +x
        "elements": [0, 1, 2, 3],
        "value": 1.0,
    }]
    soln = first_fe_code(coords, blocks, bcs, dloads, materials, block_elem_map)
    F = soln["force"]
    # Each unit-length element contributes [q*L/2, q*L/2] = [0.5, 0.5]
    np.testing.assert_allclose(F, [0.5, 1.0, 1.0, 1.0, 0.5], atol=1e-12)


def test_zero_length_element_raises():
    # Make nodes 1 and 2 colocated -> zero-length element [1,2]
    coords, blocks, bcs, dloads, materials, block_elem_map = _uniform_chain_input()
    coords = coords.copy()
    coords[2, 0] = coords[1, 0]  # x2 = x1
    import pytest
    with pytest.raises(ValueError, match="Zero-length element"):
        first_fe_code(coords, blocks, bcs, dloads, materials, block_elem_map)


def test_bad_dload_direction_raises():
    coords, blocks, bcs, dloads, materials, block_elem_map = _uniform_chain_input()
    # Dirichlet only; add a dload with zero direction (invalid)
    bcs = [{"type": DIRICHLET, "nodes": [0], "local_dof": 0, "value": 0.0}]
    dloads = [{
        "name": "bad",
        "type": "BX",
        "direction": [0],   # invalid
        "elements": [0],
        "value": 1.0,
    }]
    import pytest
    with pytest.raises(ValueError, match="direction must be ±1"):
        first_fe_code(coords, blocks, bcs, dloads, materials, block_elem_map)


def test_global_dof_mapping():
    assert global_dof(0, 0, 1) == 0
    assert global_dof(4, 0, 1) == 4
    assert global_dof(2, 0, 3) == 6  # (2 * 3) + 0
