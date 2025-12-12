import numpy as np
from wundy.elematl import NeoHookean1D
from wundy.ui import preprocess
import numpy.testing as npt
from wundy.first import newton_solve_bar1d
from wundy.schemas import DIRICHLET


def _build_mms_problem():
    L = 1.0
    a = 0.1
    A = 1.0
    mu = 10.0
    K = 0.0
    U_tip = 1.0
    mat = NeoHookean1D(mu=mu, K=K)

    def u_exact(x: float) -> float:
        xi = x / L
        return U_tip*(3.0 * xi**2 - 2.0 * xi**3)


    def strain_exact(x: float) -> float:
        xi = x / L
        du_dxi = U_tip*(6.0 * xi - 6.0 * xi**2)
        return du_dxi / L


    num_samples = 201
    x_grid = np.linspace(0.0, L, num_samples)
    sigma = np.array([mat.stress(strain_exact(x)) for x in x_grid])
    h = x_grid[1] - x_grid[0]
    dsigma_dx = np.zeros_like(sigma)
    dsigma_dx[1:-1] = (sigma[2:] - sigma[:-2]) / (2.0 * h)
    dsigma_dx[0]    = (sigma[1]  - sigma[0])   / h
    dsigma_dx[-1]   = (sigma[-1] - sigma[-2])  / h
    q_grid = -A * dsigma_dx
    xq_pairs = np.column_stack([x_grid, q_grid]).tolist()

    num_nodes = 11
    coords_1d = np.linspace(0.0, L, num_nodes)
    nodes = [[i, float(coords_1d[i])] for i in range(num_nodes)]
    elements = [[e, e, e + 1] for e in range(num_nodes - 1)]
    num_elems = len(elements)

    data = {
        "wundy": {
            "nodes": nodes,
            "elements": elements,
            "materials": [
                {
                    "name": "MAT-1",
                    "type": "neohookean",
                    "parameters": {"mu": mu, "K": K},
                    "density": 0.0,
                }
            ],
            "element blocks": [
                {
                    "name": "BLK-1",
                    "elements": list(range(num_elems)),
                    "element": {
                        "type": "T1D1",
                        "properties": {"area": A},
                    },
                    "material": "MAT-1",
                }
            ],
            "boundary conditions": [
                {"nodes": [0], "dof": 0, "type": DIRICHLET, "value": float(u_exact(0.0))},
                {"nodes": [num_nodes - 1], "dof": 0, "type": DIRICHLET, "value": float(u_exact(L))},
            ],
            "distributed loads": [
                {
                    "name": "MMS-NEO-1",
                    "elements": list(range(num_elems)),
                    "type": "BX",
                    "direction": [1.0],
                    "input_type": "table",
                    "value": xq_pairs,
                }
            ],
        }
    }

    pre = preprocess(data)
    return L, u_exact, pre


def solve_mms_neohookean_bar():
    L, u_exact, pre = _build_mms_problem()

    sol = newton_solve_bar1d(
        coords=pre["coords"],
        blocks=pre["blocks"],
        bcs=pre["bcs"],
        dload=pre.get("dload"),
        materials=pre["materials"],
        block_elem_map=pre["block_elem_map"],
        tol=1e-10,
        max_iter=25,
    )

    u_FE = sol["dofs"]
    x_nodes = pre["coords"][:, 0]
    u_exact_nodes = np.array([u_exact(x) for x in x_nodes])
    return x_nodes, u_FE, u_exact_nodes, L, u_exact


def plot_mms_neohookean_bar():
    import matplotlib.pyplot as plt

    x_nodes, u_FE, u_exact_nodes, L, u_exact = solve_mms_neohookean_bar()

    xc = np.linspace(0.0, L, 201)
    u_exact_dense = np.array([u_exact(x) for x in xc])

    plt.figure()
    plt.plot(xc, u_exact_dense, lw=2, label="Analytic Solution")
    plt.plot(x_nodes, u_FE, "k--o", lw=2, label="FE Solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def test_mms_neohookean_bar_table_load():
    x_nodes, u_FE, u_exact_nodes, L, u_exact = solve_mms_neohookean_bar()
    npt.assert_allclose(u_FE, u_exact_nodes, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
    from mms_bar import plot_mms_neohookean_bar
    plot_mms_neohookean_bar()
    # python -m mms_bar
    # to print a nice plot of the results