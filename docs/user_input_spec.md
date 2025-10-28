# 🧭 User Input Specification

> **Version:** Draft 1.0  
> **Source:** [`schemas.py`](../src/wundy/schemas.py)  
> **Purpose:** Defines the structure and validation rules for all model input files used by the *wundy* finite element analysis (FEA) framework.

All *wundy* simulations are defined by a single **YAML** or **JSON** input file that fully describes the model geometry, materials, boundary conditions, and loads.  
This document explains the format and validation rules enforced by the input schema.

---

## 📘 Top-Level Structure

Every input file must begin with a top-level key:

```yaml
wundy:
```

All model definitions are nested under this key.  
A complete input file typically includes the following sections:

```yaml
wundy:
  nodes: [...]
  elements: [...]
  boundary conditions: [...]
  materials: [...]
  element blocks: [...]
  node sets: [...]
  element sets: [...]
  concentrated loads: [...]
  distributed loads: [...]
```

Not all sections are required for every simulation, but most models will define:
- `nodes`
- `elements`
- `materials`
- `element blocks`
- `boundary conditions`

---

## 🧱 1. Nodes

Defines all nodal coordinates in the model.

```yaml
nodes:
  - [1, 0.0]
  - [2, 1.0]
  - [3, 2.0]
```

| Field | Description | Type | Example |
|--------|--------------|------|----------|
| `node_id` | Unique node number | integer | `1` |
| `x`, `y`, `z` | Node coordinates | float | `1.0` |

**Rules**
- Each inner list: `[node_id, x, (y, z)]`
- IDs must be unique integers.
- Coordinates must be numeric (`float` or `int`).
- Only `x` is required for 1D problems.

---

## 🧩 2. Elements

Defines how nodes connect to form elements.

```yaml
elements:
  - [1, 1, 2]
  - [2, 2, 3]
```

| Field | Description | Type | Example |
|--------|--------------|------|----------|
| `element_id` | Unique element number | integer | `1` |
| `connectivity` | List of node IDs | list[int] | `[1, 2]` |

**Rules**
- Each list: `[element_id, node_i, node_j]`
- Only 1D truss/bar (`T1D1`) elements are currently supported.

---

## 🧪 3. Materials

Specifies the material models and their parameters.

```yaml
materials:
  - type: ELASTIC
    name: steel
    parameters:
      E: 10.0
      nu: 0.3
    density: 1.0
```

| Field | Description | Required | Type | Notes |
|--------|--------------|-----------|------|-------|
| `type` | Material type | ✅ | string | Only `ELASTIC` supported |
| `name` | Material name | ✅ | string | Must match block reference |
| `parameters.E` | Young’s modulus | ✅ | float > 0 | |
| `parameters.nu` | Poisson’s ratio | ✅ | float in [-1, 0.5) | |
| `density` | Material density | optional | float > 0 | Must be positive |

---

## 🧱 4. Element Blocks

Groups elements by material and element type.

```yaml
element blocks:
  - name: block-1
    material: steel
    elements: [1, 2, 3]
    element:
      type: T1D1
      properties:
        area: 1.0
```

| Field | Description | Type | Rules |
|--------|--------------|------|-------|
| `name` | Block name | string | Must be unique |
| `material` | Material name | string | Must match a defined material |
| `elements` | Element IDs or set name | list[int] or str | |
| `element.type` | Element type | string | Must be valid (`T1D1`) |
| `element.properties.area` | Cross-sectional area | float > 0 | |

---

## 🧲 5. Boundary Conditions

Specifies fixed supports, prescribed displacements, or constraints.

```yaml
boundary conditions:
  - name: fixed-end
    nodes: [1]
    dof: X
    type: DIRICHLET
    value: 0.0
```

| Field | Description | Default | Type | Notes |
|--------|--------------|----------|------|-------|
| `nodes` | Node list, single node, or node set name | — | list[int] or str | |
| `dof` | Degree of freedom | `X` | str | Only `X` supported (1D) |
| `type` | Constraint type | `DIRICHLET` | str | `DIRICHLET` (displacement) or `NEUMANN` (traction) |
| `value` | Prescribed value | `0.0` | float | |

---

## 🎯 6. Concentrated Loads *(optional)*

Applies nodal forces directly to nodes.

```yaml
concentrated loads:
  - name: load-1
    nodes: [3]
    dof: X
    value: 50.0
```

| Field | Description | Default | Type |
|--------|--------------|----------|------|
| `nodes` | Node list, single node, or set | — | list[int] or str |
| `dof` | Degree of freedom | `X` | str |
| `value` | Load value | 0.0 | float |

---

## 🌊 7. Distributed Loads *(optional)*

Defines distributed or body forces applied along elements.

```yaml
distributed loads:
  - name: dload-1
    elements: [1, 2, 3]
    type: BX
    direction: [1]
    value: 10.0
```

| Field | Description | Required | Type | Notes |
|--------|--------------|-----------|------|-------|
| `elements` | Element list or element set name | ✅ | list[int] or str | |
| `type` | Load type | ✅ | str | Must be `BX` (1D body load) or `GRAV` |
| `direction` | Direction vector | ✅ | list[float] | Length = 1 for 1D |
| `value` | Magnitude of load | ✅ | float | Units: force/length |
| `name` | Label for load case | optional | str | Uppercase normalized |

---

## 🔖 8. Node Sets *(optional)*

Defines named groups of nodes for reuse in BCs or loads.

```yaml
node sets:
  - name: FIXED
    nodes: [1]
```

| Field | Description | Type |
|--------|--------------|------|
| `name` | Set name | string |
| `nodes` | Node list | list[int] |

---

## 🧩 9. Element Sets *(optional)*

Defines named groups of elements for reuse in loads or blocks.

```yaml
element sets:
  - name: ALL
    elements: [1, 2, 3, 4]
```

| Field | Description | Type |
|--------|--------------|------|
| `name` | Set name | string |
| `elements` | Element list | list[int] |

---

## 🧾 Minimal Working Example

```yaml
wundy:
  nodes: [[1, 0], [2, 1]]
  elements: [[1, 1, 2]]

  boundary conditions:
    - nodes: [1]
      dof: X
      type: DIRICHLET
      value: 0.0

  distributed loads:
    - elements: [1]
      type: BX
      direction: [1]
      value: 10.0

  materials:
    - type: ELASTIC
      name: mat-1
      parameters:
        E: 10.0
        nu: 0.3
      density: 1.0

  element blocks:
    - name: block-1
      material: mat-1
      elements: [1]
      element:
        type: T1D1
        properties:
          area: 1.0
```

---

## 🧮 Validation and Error Checking

The schema enforces type, range, and naming consistency before running any analysis.  
Validation can be run manually in Python:

```python
from schemas import input_schema
import yaml

with open("example.yaml") as f:
    data = yaml.safe_load(f)

input_schema.validate(data)
print("Validation successful!")
```

If an error occurs, the schema raises a descriptive exception:

```
SchemaError: Key 'E' must be positive (> 0)
```

---

## ⚠️ Common Errors and Solutions

| Error | Cause | Fix |
|--------|--------|-----|
| `SchemaError: Unknown element type` | `type:` misspelled or not `T1D1` | Use `T1D1` exactly |
| `SchemaError: Unknown load type` | Missing or invalid `type` in distributed load | Use `BX` or `GRAV` |
| `SchemaError: Invalid direction length` | `direction` list wrong size | Use `[1]` for 1D problems |
| `SchemaError: Density must be > 0` | Missing or zero density | Add `density: 1.0` to material |
| `KeyError: 'all'` | Used `elements: all` or `nodes: all` | Define `element sets` or `node sets` first |
| `SchemaError: nu must be between -1 and .5` | Invalid Poisson’s ratio | Keep within range |

---

## 🧰 Notes for Future Extensions

This schema is designed to expand easily to 2D and 3D element types.  
Planned extensions include:
- Support for `Q2D4`, `T3D2`, and higher-order elements  
- Multiple DOFs per node (`X`, `Y`, `Z`)  
- Directional distributed loads (e.g., `[1, 0, 0]` vectors)  
- Pressure and temperature load cases  
- Named load cases and step definitions  

---


