"""
Tests for wundy.first.first_fe_code

This suite exercises a minimal 1D axial bar/rod chain assembled from
linear 2-node elements, using a small YAML input that `wundy.ui.load`
and `wundy.ui.preprocess` transform into solver inputs. The mesh has
five nodes and four elements, with a Dirichlet boundary condition at
node 1 (fixed in x) and a single concentrated load at node 5. Material
is linear elastic and elements share constant cross-sectional area.

Goals covered by these tests:
- The assembled global stiffness matrix matches the expected tridiagonal form
  for a uniform 1D bar mesh (symmetry and values).
- The global load vector reflects the applied point load only.
- The solved nodal DOFs match the expected linear displacement field.
- Structural sanity checks: matrix sizes, symmetry, and positive definiteness.
- Edge case: invalid inputs (e.g., zero area) raise an error.
- Future work placeholder: uniform distributed load (currently expected to fail).

Assumptions:
- `wundy.ui.load` accepts a file-like stream with YAML content.
- `wundy.ui.preprocess` returns a dict with keys:
  "coords", "blocks", "bcs", "dload", "materials", "block_elem_map".
- `wundy.first.first_fe_code` returns a dict with keys:
  "dofs", "stiff", "force".

If your local schema differs, adjust the YAML keys or expectations as needed.
"""

import io
import numpy as np
import pytest

import wundy
import wundy.first


def _run_model(yaml_text: str):
    """Helper to load, preprocess, and solve a small YAML model."""
    file = io.StringIO(yaml_text)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)
    return wundy.first.first_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )


def test_first_1():
    """
    Uniform 1D chain: 5 nodes @ x = 0..4 (unit spacing), E=10, A=1,
    node 1 fixed, 2.0 point load at node 5.

    Expected:
      - Displacements: [0, 0.2, 0.4, 0.6, 0.8]
      - Global load vector: [0, 0, 0, 0, 2]
      - Global K (5x5) tridiagonal with 10 at the ends, 20 on interior diags,
        and -10 on off-diagonals adjacent to the main diagonal.
    """
    yaml_text = """\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  concentrated loads:
  - name: cload-1
    nodes: [5]
    value: 2.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: t1d1
      properties:
        area: 1
"""
    soln = _run_model(yaml_text)

    dofs = np.asarray(soln["dofs"], dtype=float)
    K = np.asarray(soln["stiff"], dtype=float)
    F = np.asarray(soln["force"], dtype=float)

    np.testing.assert_allclose(dofs, [0, 0.2, 0.4, 0.6, 0.8], rtol=0, atol=1e-12)
    np.testing.assert_allclose(F, [0, 0, 0, 0, 2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        K,
        [
            [10, -10,   0,   0,   0],
            [-10, 20, -10,   0,   0],
            [  0, -10,  20, -10,  0],
            [  0,   0, -10,  20, -10],
            [  0,   0,   0, -10, 10],
        ],
        rtol=0,
        atol=1e-12,
    )


def test_stiffness_symmetry_and_shapes():
    """Basic structure checks: K is square/symmetric; F and DOFs match node count."""
    yaml_text = """\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  concentrated loads:
  - name: cload-1
    nodes: [5]
    value: 2.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: t1d1
      properties:
        area: 1
"""
    soln = _run_model(yaml_text)
    dofs = np.asarray(soln["dofs"], dtype=float)
    K = np.asarray(soln["stiff"], dtype=float)
    F = np.asarray(soln["force"], dtype=float)

    n = dofs.size
    assert K.shape == (n, n), "Stiffness must be square and match DOF count"
    assert F.shape == (n,), "Force vector length must match DOF count"
    np.testing.assert_allclose(K, K.T, atol=1e-12)


def test_stiffness_positive_definite_for_this_case():
    """
    For the fully assembled, constrained system in this test problem,
    K should be positive definite (all eigenvalues > 0).
    """
    yaml_text = """\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  concentrated loads:
  - name: cload-1
    nodes: [5]
    value: 2.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: t1d1
      properties:
        area: 1
"""
    soln = _run_model(yaml_text)
    K = np.asarray(soln["stiff"], dtype=float)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals > 0), f"Expected SPD; got min eigenvalue {eigvals.min():.3e}"


def test_zero_area_raises_error():
    """Zero cross-section area is physically invalid; solver should reject it."""
    yaml_text = """\
wundy:
  nodes: [[1, 0], [2, 1]]
  elements: [[1, 1, 2]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: t1d1
      properties:
        area: 0
"""
    with pytest.raises(Exception):
        _ = _run_model(yaml_text)


@pytest.mark.xfail(reason="Uniform distributed load not yet implemented", strict=False)
def test_first_2():
    """
    Placeholder: prescribe a uniform distributed load q over the domain and
    verify the assembled global force vector matches the expected contributions
    of linear 1D shape functions (each element of unit length with q=1
    contributes [0.5, 0.5] to its end nodes).

    Once implemented, the expected global F for 4 unit-length elements is:
      [0.5, 1.0, 1.0, 1.0, 0.5]
    """
    yaml_text = """\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  distributed loads:
  - name: dload-1
    elements: all
    value: 1.0   # units: force/length
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: t1d1
      properties:
        area: 1
"""
    soln = _run_model(yaml_text)
    F = np.asarray(soln["force"], dtype=float)
    expected = np.array([0.5, 1.0, 1.0, 1.0, 0.5], dtype=float)
    np.testing.assert_allclose(F, expected, atol=1e-12)
