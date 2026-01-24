from typing import Any

from schema import And
from schema import Optional
from schema import Or
from schema import Schema
from schema import Use

NEUMANN = 0
DIRICHLET = 1


element_types = {"T1D1"}
bc_types = {"DIRICHLET", "NEUMANN"}


def node_freedom_table(elem_type: str) -> tuple[int, ...]:
    if normalize_case(elem_type) == "T1D1":
        return (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    raise ValueError(f"Unknown element type {elem_type!r}")


def valid_element_type(name: str) -> bool:
    return normalize_case(name) in element_types


def isnumeric(x) -> bool:
    return isinstance(x, (int, float))


def ispositive(arg: float | int) -> bool:
    return arg > 0

def _validate_profile_spec(
    spec: object,
    *,
    name: str,
    default_input_type: str = "SCALAR",
    require_positive: bool = True,
) -> object:
    """
    Accept either a numeric scalar or a dict:
      {"input_type": "SCALAR"|"TABLE"|"EQUATION", "value": ...}

    TABLE expects [[x0, y0], [x1, y1], ...]
    EQUATION expects a string expression in variable x (and possibly L, np, pi).
    """
    # Backward compatible scalar
    if isnumeric(spec):
        y = float(spec)
        if require_positive and y <= 0.0:
            raise ValueError(f"{name}: must be > 0")
        return y

    if not isinstance(spec, dict):
        raise ValueError(
            f"{name}: must be a number or a dict with keys {{input_type, value}}, got {type(spec).__name__}"
        )

    itype = normalize_case(spec.get("input_type", default_input_type))
    val = spec.get("value", None)

    if itype == "SCALAR":
        if not isnumeric(val):
            raise ValueError(f"{name}: for input_type='SCALAR', value must be numeric")
        y = float(val)
        if require_positive and y <= 0.0:
            raise ValueError(f"{name}: must be > 0")
        # normalize stored form (optional; keep dict shape)
        spec["input_type"] = itype
        spec["value"] = y
        return spec

    if itype == "TABLE":
        if not isinstance(val, list):
            raise ValueError(f"{name}: for input_type='TABLE', value must be a list of [x, y] pairs")
        table: list[list[float]] = []
        for row in val:
            if not (isinstance(row, (list, tuple)) and len(row) == 2):
                raise ValueError(f"{name}: each table row must be [x, y], got {row!r}")
            x, y = row
            if not isnumeric(x) or not isnumeric(y):
                raise ValueError(f"{name}: table entries must be numeric, got {row!r}")
            y = float(y)
            if require_positive and y <= 0.0:
                raise ValueError(f"{name}: table values must be > 0, got {row!r}")
            table.append([float(x), y])
        spec["input_type"] = itype
        spec["value"] = table
        return spec

    if itype == "EQUATION":
        if not isinstance(val, str):
            raise ValueError(f"{name}: for input_type='EQUATION', value must be a string expression")
        spec["input_type"] = itype
        # leave value as-is (string)
        return spec

    raise ValueError(f"{name}: unknown input_type {itype!r}")

def list_of_type(sequence: list, type) -> bool:
    return all(isinstance(n, type) for n in sequence)


def list_of_numeric(sequence: list) -> bool:
    return all(isinstance(x, (float, int)) for x in sequence)


def list_of_int(sequence) -> bool:
    return list_of_type(sequence, int)


def list_of_list(sequence) -> bool:
    return list_of_type(sequence, list)


def normalize_case(string: str) -> str:
    return string.upper()


def dof_id_to_enum(dof: str) -> int:
    return {"X": 0, "Y": 1, "Z": 2}[normalize_case(dof)]


def valid_bc_type(arg: str) -> bool:
    return normalize_case(arg) in bc_types


def bc_type_to_enum(bc_type: str) -> int:
    return {"DIRICHLET": DIRICHLET, "NEUMANN": NEUMANN}[normalize_case(bc_type)]


def valid_dof_id(dof: str):
    # extension to 2/3D: allow dof to be xyz
    return normalize_case(dof) in {"X"}


def valid_dload_type(arg: str):
    # extension to 2/3D: allow other DLOADs
    return normalize_case(arg) in {"BX", "GRAV"}

input_types = {"SCALAR", "TABLE", "EQUATION"}


def valid_dload_input_type(arg: str) -> bool:
    return normalize_case(arg) in input_types


def validate_element(elem: dict[str, Any]) -> bool:
    if normalize_case(elem["type"]) == "T1D1":
        props = elem.get("properties", {})
        if props is None:
            props = {}
            elem["properties"] = props
        if "area" not in props:
            props["area"] = 1.0

        props["area"] = _validate_profile_spec(props["area"], name="element.properties.area", require_positive=True)
        return True
    raise ValueError(f"Unknown element type {elem['type']!r}")


def validate_material_parameters(material: dict[str, dict[str, Any]]) -> bool:
    elastic = Schema(
        {
            "E": And(isnumeric, ispositive, error="E must be > 0"),
            "nu": And(isnumeric, lambda x: -1.0 <= x < 0.5, error="nu must be between -1 and .5"),
        }
    )
    if normalize_case(material["type"]) == "ELASTIC":
        elastic.validate(material["parameters"])
    else:
        raise ValueError(f"Unknown material {material['type']!r}")
    return True

def validate_dload(dl: dict[str, Any]) -> bool:
    """
    Post-process and validate a distributed load entry based on input_type.

    input_type:
      - SCALAR   : value must be numeric -> float
      - TABLE    : value must be [[x0, q0], [x1, q1], ...]
      - EQUATION : value must be a string (e.g., "sin(pi*x/L)")
    """
    itype = normalize_case(dl.get("input_type", "SCALAR"))
    val = dl["value"]

    if itype == "SCALAR":
        # Backward-compatible: old inputs that had just a number
        if not isnumeric(val):
            raise ValueError(
                f"dload {dl.get('name', '')!r}: for input_type='SCALAR', "
                f"value must be numeric, got {type(val).__name__}"
            )
        dl["value"] = float(val)

    elif itype == "TABLE":
        # Expect list of [x, q] pairs
        if not isinstance(val, list):
            raise ValueError(
                f"dload {dl.get('name', '')!r}: for input_type='TABLE', "
                f"value must be a list of [x, q] pairs"
            )

        table: list[list[float]] = []
        for row in val:
            if not (isinstance(row, (list, tuple)) and len(row) == 2):
                raise ValueError(
                    f"dload {dl.get('name', '')!r}: each table row must be [x, q], "
                    f"got {row!r}"
                )
            x, q = row
            if not isnumeric(x) or not isnumeric(q):
                raise ValueError(
                    f"dload {dl.get('name', '')!r}: table entries must be numeric, "
                    f"got {row!r}"
                )
            table.append([float(x), float(q)])

        dl["value"] = table

    elif itype == "EQUATION":
        # Expect a string expression, e.g. "sin(pi*x/L)"
        if not isinstance(val, str):
            raise ValueError(
                f"dload {dl.get('name', '')!r}: for input_type='EQUATION', "
                f"value must be a string expression, got {type(val).__name__}"
            )
        # Leave string as-is; later code (preprocessor / element force) will eval/parse it

    else:
        # This should not happen if the schema input_type validator is correct
        raise ValueError(f"Unknown dload input_type {itype!r}")

    return True

nodes_schema = Schema(
    And(
        list,
        list_of_list,
        lambda outer: all(isinstance(inner[0], int) for inner in outer),  # node label
        lambda outer: all(isinstance(f, (int, float)) for inner in outer for f in inner[1:]),
        Use(lambda outer: [[int(inner[0]), *[float(_) for _ in inner[1:]]] for inner in outer]),
    )
)

elements_schema = Schema(
    And(
        list,
        list_of_list,
        lambda outer: all(list_of_int(inner) for inner in outer),
    )
)

nset_schema = Schema(
    {
        "name": And(str, Use(normalize_case)),
        "nodes": And(list, list_of_int),
    },
)

elset_schema = Schema(
    {
        "name": And(str, Use(normalize_case)),
        "elements": And(list, list_of_int),
    },
)

boundary_schema = Schema(
    And(
        {
            "nodes": Or(
                And(str, Use(normalize_case)),  # node set name
                And(int, Use(lambda n: [n])),  # single node
                And(list, list_of_int),  # list of nodes
            ),
            Optional("dof", default=0): And(str, valid_dof_id, Use(dof_id_to_enum)),
            Optional("name"): And(str, Use(normalize_case)),
            Optional("value", default=0.0): And(isnumeric, Use(float)),
            Optional("type", default=DIRICHLET): And(str, valid_bc_type, Use(bc_type_to_enum)),
        },
    )
)

cload_schema = Schema(
    And(
        {
            "nodes": Or(
                And(str, Use(normalize_case)),  # node set name
                And(int, Use(lambda n: [n])),  # single node
                And(list, list_of_int),  # list of nodes
            ),
            Optional("dof", default=0): And(str, valid_dof_id, Use(dof_id_to_enum)),
            Optional("name"): And(str, Use(normalize_case)),
            Optional("value", default=0.0): Use(float),
        },
    )
)

dload_schema = Schema(
    And(
        {
            "elements": Or(
                And(str, Use(normalize_case)),  # element set name
                And(int, Use(lambda e: [e])),  # single element
                And(list, list_of_int),        # list of elements
            ),
            "type": And(str, valid_dload_type, Use(normalize_case)),
            # Allow any type initially; validate_dload() will refine it
            "value": object,
            "direction": And(
                list,
                list_of_numeric,
                lambda sequence: len(sequence) == 1,  # change to <= 2/3 for 2D/3D
                Use(lambda sequence: [float(x) for x in sequence]),
            ),
            Optional("name"): And(str, Use(normalize_case)),
            Optional("input_type", default="SCALAR"): And(
                str,
                valid_dload_input_type,
                Use(normalize_case),
            ),
        },
        lambda d: validate_dload(d),  # enforce value shape based on input_type
    )
)

material_schema = Schema(
    And(
        {
            "type": And(str, Use(normalize_case)),
            "name": And(str, Use(normalize_case)),
            "parameters": {str: object},
            Optional("density", default=0.0): And(isnumeric, ispositive),
        },
        lambda d: validate_material_parameters(d),
    )
)

block_schema = Schema(
    And(
        {
            "name": And(str, Use(normalize_case)),
            "material": And(str, Use(normalize_case)),
            "elements": Or(
                And(str, Use(normalize_case)),
                And(list, list_of_int),
            ),
            "element": {
                "type": And(str, valid_element_type, Use(normalize_case)),
                Optional("properties", default=dict()): {str: object},
            },
        },
        lambda d: validate_element(d["element"]),
    )
)

input_schema = Schema(
    {
        "wundy": {
            "nodes": nodes_schema,
            "elements": elements_schema,
            "boundary conditions": [boundary_schema],
            "materials": [material_schema],
            "element blocks": [block_schema],
            Optional("node sets"): [nset_schema],
            Optional("element sets"): [elset_schema],
            Optional("concentrated loads"): [cload_schema],
            Optional("distributed loads"): [dload_schema],
        }
    }
)
