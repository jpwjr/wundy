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
from wundy.schemas import DIRICHLET, NEUMANN
import wundy.elematl


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


def test_first_2():
    """
    Uniform 1D chain with distributed load:
      - Nodes at x = 0..4 (unit spacing), E=10, A=1
      - Node 1 fixed
      - Uniform distributed load q = 10 (force/length) on all elements

    Expected:
      - Consistent global load vector: [5, 10, 10, 10, 5]
      - Displacements (analytic and FE agree): [0, 3.5, 6.0, 7.5, 8.0]
      - Global K identical to test_first_1 (tridiagonal form)
    """
    yaml_text = """\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]

  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
    type: dirichlet
    value: 0.0

  distributed loads:
  - name: dload-1
    elements: [1, 2, 3, 4]  
    type: BX              
    direction: [1]          
    value: 10.0             

  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
    density: 1.0             

  element blocks:
  - material: mat-1
    name: block-1
    elements: [1, 2, 3, 4]    
    element:
      type: t1d1
      properties:
        area: 1
"""
    soln = _run_model(yaml_text)

    dofs = np.asarray(soln["dofs"], dtype=float)
    K = np.asarray(soln["stiff"], dtype=float)
    F = np.asarray(soln["force"], dtype=float)

    # Expected consistent nodal forces for q=10 over 4 unit elements:
    np.testing.assert_allclose(F, [5.0, 10.0, 10.0, 10.0, 5.0], atol=1e-12)

    # Same K pattern as test_first_1
    np.testing.assert_allclose(
        K,
        [
            [10, -10,   0,   0,   0],
            [-10, 20, -10,   0,   0],
            [  0, -10,  20, -10,  0],
            [  0,   0, -10,  20, -10],
            [  0,   0,   0, -10, 10],
        ],
        atol=1e-12,
        rtol=0,
    )

    # Expected displacements for fixed x=0 end and no end point load:
    # u(x) = -x^2/2 + 4x  => [0, 3.5, 6.0, 7.5, 8.0] at x=0..4
    np.testing.assert_allclose(dofs, [0.0, 3.5, 6.0, 7.5, 8.0], atol=1e-10)

#Test to ensure Gauss quadrature matches analytical integration for 1D linear bar
def test_element_gauss_matches_analytical_linear_bar_with_diagnostics():

    import numpy as np
    import wundy.elematl as elematl

    # Parameters
    E = 10.0
    A = 2.0
    L = 3.0
    q0 = 4.0
    TOL = 1e-12

    # Material and element coordinates (forward and reversed)
    mat = elematl.LinearElastic1D(E=E)
    xe_fwd = np.array([0.0, L], dtype=float)
    xe_rev = np.array([L, 0.0], dtype=float)

    # --- Expected analytical results (forward orientation) ---
    Ke_expected = (E * A / L) * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    f_expected = q0 * L * np.array([0.5, 0.5], dtype=float)

    # --- Compute Gauss-based results ---
    Ke_fwd = elematl.element_stiffness_bar1d(xe_fwd, A, mat, ue=None, n_gauss=2)
    Ke_rev = elematl.element_stiffness_bar1d(xe_rev, A, mat, ue=None, n_gauss=2)
    f_fwd = elematl.element_external_body_bar1d(xe_fwd, q0, n_gauss=2)
    f_rev = elematl.element_external_body_bar1d(xe_rev, q0, n_gauss=2)

    # --- Shape and finiteness checks (stiffness) ---
    assert Ke_fwd.shape == (2, 2), (
        f"Ke_fwd shape incorrect; expected (2,2), got {Ke_fwd.shape}"
    )
    assert Ke_rev.shape == (2, 2), (
        f"Ke_rev shape incorrect; expected (2,2), got {Ke_rev.shape}"
    )
    assert np.isfinite(Ke_fwd).all(), f"Ke_fwd contains non-finite values:\n{Ke_fwd}"
    assert np.isfinite(Ke_rev).all(), f"Ke_rev contains non-finite values:\n{Ke_rev}"

    # --- Symmetry checks (stiffness) ---
    assert np.allclose(Ke_fwd, Ke_fwd.T, atol=TOL), (
        f"Ke_fwd is not symmetric within atol={TOL}:\nKe_fwd=\n{Ke_fwd}\nKe_fwd.T=\n{Ke_fwd.T}"
    )
    assert np.allclose(Ke_rev, Ke_rev.T, atol=TOL), (
        f"Ke_rev is not symmetric within atol={TOL}:\nKe_rev=\n{Ke_rev}\nKe_rev.T=\n{Ke_rev.T}"
    )

    # --- Analytical vs. Gauss (stiffness) ---
    assert np.allclose(Ke_fwd, Ke_expected, atol=TOL, rtol=0), (
        "Gauss stiffness (forward order) does not match analytical result.\n"
        f"Params: E={E}, A={A}, L={L}\n"
        f"Expected:\n{Ke_expected}\nGot:\n{Ke_fwd}\n"
        f"Abs diff:\n{np.abs(Ke_fwd - Ke_expected)}"
    )
    assert np.allclose(Ke_rev, Ke_expected, atol=TOL, rtol=0), (
        "Gauss stiffness (reversed order) does not match analytical result.\n"
        f"Params: E={E}, A={A}, L={L}\n"
        f"Expected:\n{Ke_expected}\nGot:\n{Ke_rev}\n"
        f"Abs diff:\n{np.abs(Ke_rev - Ke_expected)}"
    )

    # --- Consistent body load: forward orientation only ---
    assert f_fwd.shape == (2,), f"f_fwd shape incorrect; expected (2,), got {f_fwd.shape}"
    assert np.isfinite(f_fwd).all(), f"f_fwd contains non-finite values: {f_fwd}"
    assert np.allclose(f_fwd, f_expected, atol=TOL, rtol=0), (
        "Gauss consistent nodal load (forward order) does not match analytical result.\n"
        f"Params: q0={q0}, L={L}\n"
        f"Expected: {f_expected}\nGot: {f_fwd}\n"
        f"Abs diff: {np.abs(f_fwd - f_expected)}"
    )

    # --- Document current behavior for reversed order (signed J) ---
    # With xe_rev (nodes reversed), your implementation uses J = (x2-x1)/2 < 0,
    # so the Gauss-integrated body load flips sign. Keep this explicit so a future
    # change to |J| will cause a helpful failure message here.
    assert np.allclose(f_rev, f_expected, atol=TOL, rtol=0), (
        "For reversed node order, element_external_body_bar1d currently integrates "
        "with a signed Jacobian (J<0), so the body-load vector flips sign.\n"
        "This assertion documents that behavior. If you intentionally switch to "
        "using |J| for body loads, update this assertion to expect +f_expected.\n"
        f"Expected (with signed J): {-f_expected}\nGot: {f_rev}"
    )

    # --- Invariance of Ke to node order (explicit) ---
    assert np.allclose(Ke_fwd, Ke_rev, atol=TOL, rtol=0), (
        "Ke changes when node order is reversed; Jacobian/kinematics may be wrong.\n"
        f"Ke_fwd:\n{Ke_fwd}\nKe_rev:\n{Ke_rev}\n"
        f"Abs diff:\n{np.abs(Ke_fwd - Ke_rev)}"
    )
    assert np.allclose(f_fwd, f_rev, atol=TOL, rtol=0), (
        "Consistent nodal loads change when node order is reversed; shape/Jacobian may be wrong.\n"
        f"f_fwd: {f_fwd}\n"
        f"f_rev: {f_rev}\n"
        f"Abs diff: {np.abs(f_fwd - f_rev)}"
    )
def test_newton_linear_converges_in_one_step():

    coords = np.array([[0.0], [1.0], [2.0]], dtype=float)

    blocks = [
        {
            "id": 1,
            "material": "mat_lin",
            "element": {
                "type": "T1D1",
                "properties": {"area": 1.0},
            },
          
            "connect": np.array([[0, 1], [1, 2]], dtype=int),
        }
    ]

    materials = {
        "mat_lin": {
            "type": "linear_elastic",
            "parameters": {"E": 10.0},
            "density": 0.0,
        }
    }

    bcs = [
        {
            "type": DIRICHLET,
            "nodes": [0],
            "local_dof": 0,
            "value": 0.0,
        },
        {
            "type": NEUMANN,
            "nodes": [2],
            "local_dof": 0,
            "value": 10.0,
        },
    ]

    dload = None
    block_elem_map: dict[int, tuple[int, int]] = {}

    lin_sol = wundy.first.first_fe_code(
        coords=coords,
        blocks=blocks,
        bcs=bcs,
        dload=dload,
        materials=materials,
        block_elem_map=block_elem_map,
    )
    u_lin = lin_sol["dofs"]

    newton_sol = wundy.first.newton_solve_bar1d(
        coords=coords,
        blocks=blocks,
        bcs=bcs,
        dload=dload,
        materials=materials,
        block_elem_map=block_elem_map,
        tol=1e-12,
        max_iter=2, # checking for single step convergence, has to run two steps since it is looking at delta u 
        # if u after step 1 == u after step 2 then it converged in one step
    )
    u_newton = newton_sol["dofs"]

    assert np.allclose(u_newton, u_lin, rtol=1e-12, atol=1e-12)

def test_dload_constant_table_matches_scalar():

  yaml_scalar = """\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
    type: dirichlet
    value: 0.0

  distributed loads:
  - name: dload-const-scalar
    elements: [1, 2, 3, 4]
    type: BX
    direction: [1]
    value: 10.0

  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
    density: 1.0

  element blocks:
  - material: mat-1
    name: block-1
    elements: [1, 2, 3, 4]
    element:
      type: t1d1
      properties:
        area: 1
"""
  yaml_table = """\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]

  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
    type: dirichlet
    value: 0.0

  distributed loads:
  - name: const-table
    elements: [1, 2, 3, 4]
    type: BX
    direction: [1]
    input_type: table
    value: [[0.0, 10.0], [4.0, 10.0]]

  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
    density: 1.0

  element blocks:
  - material: mat-1
    name: block-1
    elements: [1, 2, 3, 4]
    element:
      type: t1d1
      properties:
        area: 1
"""
  soln_scalar = _run_model(yaml_scalar)
  soln_table = _run_model(yaml_table)

  F_scalar = np.asarray(soln_scalar["force"], dtype=float)
  F_table = np.asarray(soln_table["force"], dtype=float)
  u_scalar = np.asarray(soln_scalar["dofs"], dtype=float)
  u_table = np.asarray(soln_table["dofs"], dtype=float)

    # Identical global forces and displacements
  np.testing.assert_allclose(F_table, F_scalar, rtol=0.0, atol=1e-12)
  np.testing.assert_allclose(u_table, u_scalar, rtol=0.0, atol=1e-12)


def test_arbitrary_dload_equation_qx_single_element():
    """
    Single 2-node bar on [0, 1], E=1, A=1, q(x) = x, both ends fixed.

    Analytic consistent nodal loads for linear shape functions:
      N1 = 1 - x, N2 = x, q(x) = x:

      f1 = ∫_0^1 N1 * x dx = ∫_0^1 (x - x^2) dx = 1/6
      f2 = ∫_0^1 N2 * x dx = ∫_0^1 x^2 dx       = 1/3
    """
    yaml_text = """\
wundy:
  nodes: [[1, 0.0], [2, 1.0]]
  elements: [[1, 1, 2]]

  boundary conditions:
  - name: both-fixed
    dof: x
    nodes: [1, 2]
    type: dirichlet
    value: 0.0

  distributed loads:
  - name: qx-equation
    elements: [1]
    type: BX
    direction: [1]
    input_type: equation
    value: "x"

  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 1.0
      nu: 0.3
    density: 1.0

  element blocks:
  - material: mat-1
    name: block-1
    elements: [1]
    element:
      type: t1d1
      properties:
        area: 1.0
"""
    soln = _run_model(yaml_text)
    F = np.asarray(soln["force"], dtype=float)

    f_expected = np.array([1.0 / 6.0, 1.0 / 3.0], dtype=float)

    np.testing.assert_allclose(F, f_expected, rtol=1e-12, atol=1e-12)


def test_dload_table_vs_equation_sine_profile():
    """
    Non-uniform q(x) ≈ sin(pi*x/L) over 4 elements on [0, 4].

    The same physical load is described in two ways:
      - TABLE: dense [x, q(x)] sampling
      - EQUATION: expr = "np.sin(pi*x/L)"

    The assembled global forces should agree to within interpolation/quadrature
    tolerance.
    """
    yaml_table = """\
wundy:
  nodes: [[1, 0.0], [2, 1.0], [3, 2.0], [4, 3.0], [5, 4.0]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]

  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
    type: dirichlet
    value: 0.0

  distributed loads:
  - name: dload-sine-table
    elements: [1, 2, 3, 4]
    type: BX
    direction: [1]
    input_type: table
    value: [
      [0.0, 0.0],
      [0.5, 0.3826834323650898],
      [1.0, 0.7071067811865476],
      [1.5, 0.9238795325112866],
      [2.0, 1.0],
      [2.5, 0.9238795325112867],
      [3.0, 0.7071067811865477],
      [3.5, 0.3826834323650899],
      [4.0, 1.2246467991473532e-16]
    ]

  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
    density: 1.0

  element blocks:
  - material: mat-1
    name: block-1
    elements: [1, 2, 3, 4]
    element:
      type: t1d1
      properties:
        area: 1.0
"""
    yaml_equation = """\
wundy:
  nodes: [[1, 0.0], [2, 1.0], [3, 2.0], [4, 3.0], [5, 4.0]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]

  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
    type: dirichlet
    value: 0.0

  distributed loads:
  - name: dload-sine-equation
    elements: [1, 2, 3, 4]
    type: BX
    direction: [1]
    input_type: equation
    value: "np.sin(pi*x/L)"

  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
    density: 1.0

  element blocks:
  - material: mat-1
    name: block-1
    elements: [1, 2, 3, 4]
    element:
      type: t1d1
      properties:
        area: 1.0
"""
    soln_table = _run_model(yaml_table)
    soln_equation = _run_model(yaml_equation)

    F_table = np.asarray(soln_table["force"], dtype=float)
    F_equation = np.asarray(soln_equation["force"], dtype=float)

    np.testing.assert_allclose(F_equation, F_table, rtol=2e-2, atol=1e-3)


from wundy.beam import first_beam_code, assemble_beam_chain

def test_beam_element_stiffness_and_shape():
    """
    Basic sanity: assembled K is square, symmetric, and matches 2 DOFs/node.
    """
    L = 1.0
    E = 10.0
    I = 2.0
    q = 0.0

    n_elem = 3
    n_node = n_elem + 1

    coords = np.zeros((n_node, 1), dtype=float)
    coords[:, 0] = np.linspace(0.0, L, n_node)

    connect = np.array([[i, i + 1] for i in range(n_elem)], dtype=int)


    K, F = assemble_beam_chain(coords, connect, E, I, q)

    n_dof = n_node * 2
    assert K.shape == (n_dof, n_dof)
    assert F.shape == (n_dof,)

    npt.assert_allclose(K, K.T, atol=1e-12)


def test_cantilever_beam_uniform_load_tip_deflection():
    """
    Euler–Bernoulli cantilever beam, length L, uniform load q:

        w(L) = q L^4 / (8 E I)

    Model: 4 Hermite beam elements, fixed at x=0 (w=0, θ=0), free at x=L.
    """
    L = 1.0
    E = 10.0
    I = 2.0
    q = 1.0  

    n_elem = 4
    n_node = n_elem + 1

    coords = np.zeros((n_node, 1), dtype=float)
    coords[:, 0] = np.linspace(0.0, L, n_node)

    connect = np.array([[i, i + 1] for i in range(n_elem)], dtype=int)

    bcs = [
        {"type": DIRICHLET, "node": 0, "local_dof": 0, "value": 0.0}, 
        {"type": DIRICHLET, "node": 0, "local_dof": 1, "value": 0.0}, 
    ]

    sol = first_beam_code(
        coords=coords,
        connect=connect,
        E=E,
        I=I,
        q=q,
        bcs=bcs,
    )

    u = sol["dofs"]
    K = sol["stiff"]
    F = sol["force"]

    dof_per_node = 2
    tip_node = n_node - 1
    tip_dof = tip_node * dof_per_node + 0
    w_tip_FE = u[tip_dof]

    w_tip_exact = q * L**4 / (8.0 * E * I)

    npt.assert_allclose(w_tip_FE, w_tip_exact, rtol=5e-2, atol=1e-4)

    npt.assert_allclose(K, K.T, atol=1e-12)

    assert np.linalg.norm(F) > 0.0