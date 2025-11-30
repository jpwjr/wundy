import logging
from typing import IO
from typing import Any

import numpy as np
import yaml

from .schemas import NEUMANN
from .schemas import input_schema
from .elematl import make_q_constant, make_q_from_table, make_q_from_equation


logger = logging.getLogger(__name__)


def load(file: IO[Any]) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(file)
    return input_schema.validate(data)


def set_element_defaults(elem: dict[str, Any]) -> bool:
    if elem["type"].upper() == "T1D1":
        nft = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        props = {"node_per_elem": 2, "freedom_table": [nft, nft]}
        elem["properties"].update(props)
    else:
        raise ValueError(f"Unknown element type {elem['type']!r}")
    return True


def unique_name(named_items: list[dict], stem: str) -> str:
    names = [item.get("name") for item in named_items]
    i = 1
    while True:
        name = f"{stem.upper()}-{i}"
        if name not in names:
            return name
        i += 1


def preprocess(data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Preprocess and transform user input.

    Assumptions: User input was loaded and validated by ``load``

    """
    errors: int = 0

    inp = data["wundy"]

    preprocessed: dict[str, Any] = {}

    num_node: int = len(inp["nodes"])
    max_dim: int = max(len(n[1:]) for n in inp["nodes"])
    node_map: dict[int, int] = preprocessed.setdefault("node_map", {})
    coords = preprocessed["coords"] = np.zeros((num_node, max_dim))
    for i, node in enumerate(inp["nodes"]):
        nid, *xc = node
        node_map[nid] = i
        coords[i, : len(xc)] = xc

    num_elem: int = len(inp["elements"])
    elem_map: dict[int, int] = preprocessed.setdefault("elem_map", {})
    for i, element in enumerate(inp["elements"]):
        elem_map[element[0]] = i

    # Put node sets in dictionary for easier look up
    nsets: dict[str, Any] = preprocessed.setdefault("nsets", {})
    nsets["all"] = list(range(num_node))
    for ns in inp.get("node sets", []):
        name = ns["name"]
        if name in nsets:
            errors += 1
            logger.error(f"Duplicate node set {name!r}")
        else:
            nodes: list[int] = []
            for n in ns["nodes"]:
                if n not in node_map:
                    errors += 1
                    logger.error(f"Node {n} in node set {name} is not defined")
                else:
                    nodes.append(node_map[n])
            nsets[name] = nodes

    # Put element sets in dictionary for easier look up
    elsets: dict[str, Any] = preprocessed.setdefault("elsets", {})
    elsets["ALL"] = list(range(num_elem))
    for es in inp.get("element sets", []):
        name = es["name"]
        if name in elsets:
            errors += 1
            logger.error(f"Duplicate element set {name!r}")
        else:
            elems: list[int] = []
            for e in es["elements"]:
                if e not in elem_map:
                    errors += 1
                    logger.error(f"Element {e} in element set {name} is not defined")
                else:
                    elems.append(elem_map[e])
            elsets[name] = elems

    # Put materials in dictionary for easier look up
    materials: dict[str, Any] = preprocessed.setdefault("materials", {})
    for material in inp["materials"]:
        name = material["name"]
        if name in materials:
            errors += 1
            logger.error(f"Duplicate material {name!r}")
        else:
            materials[name] = {"type": material["type"], "parameters": material["parameters"], "density": material["density"],}

    # Put element blocks in dictionary for easier look up
    blocks: list[Any] = preprocessed.setdefault("blocks", [])
    for eb in inp["element blocks"]:
        name = eb["name"]
        if name in blocks:
            errors += 1
            logger.error(f"Duplicate element block {name!r}")
            continue
        if eb["material"] not in materials:
            errors += 1
            logger.error(
                f"material {eb['material']!r}, required by element block {name}, not defined"
            )
            continue
        block: dict[str, Any] = {}
        block["name"] = name
        block["element"] = eb["element"]
        set_element_defaults(block["element"])
        block["material"] = eb["material"]
        elems: list[int] = []
        if isinstance(eb["elements"], str):
            # elements given as set name
            if eb["elements"] not in elsets:
                errors += 1
                logger.error(
                    f"element set {eb['elements']!r}, required by element block {name}, not defined"
                )
                continue
            elems.extend(elsets[eb["elements"]])
        else:
            for e in eb["elements"]:
                if e not in elem_map:
                    errors += 1
                    logger.error(f"Element {e}, required for element block {name}, is not defined")
                else:
                    elems.append(elem_map[e])
        if not elems:
            errors += 1
            logger.error(f"No elements defined for element block {name}")
        else:
            connect: list[list[int]] = []
            for e in elems:
                eid, *nodes = inp["elements"][e]
                if connect and len(nodes) != len(connect[0]):
                    errors += 1
                    logger.error(
                        f"Inconsistent element connectivity in element block {name}. "
                        "(All elements must have the same number of nodes)"
                    )
                    break
                row: list[int] = []
                for n in nodes:
                    if n not in node_map:
                        errors += 1
                        logger.error(
                            f"Node {n} of element {eid} in element block {name} is not in node map"
                        )
                    else:
                        row.append(node_map[n])
                connect.append(row)
            else:
                block["connect"] = np.array(connect, dtype=int)
                # Map from global index to local index
                block["elem_map"] = dict(zip(elems, range(len(elems))))
                blocks.append(block)

    # Convert boundary conditions to tags/vals that can be used by the assembler
    boundary: list[Any] = preprocessed.setdefault("bcs", [])
    for i, bc in enumerate(inp["boundary conditions"]):
        if "name" in bc:
            name = bc["name"]
        else:
            name = unique_name(inp["boundary conditions"], "BOUNDARY")
            bc["name"] = name
        nodes: list[int] = []
        if isinstance(bc["nodes"], str):
            if bc["nodes"] not in nsets:
                errors += 1
                logger.error(
                    f"Nodeset {bc['nodes']}, required by boundary condition {i + 1}, is not defined"
                )
            else:
                nodes.extend(nsets[bc["nodes"]])
        else:
            for n in bc["nodes"]:
                if n not in node_map:
                    errors += 1
                    logger.error(
                        f"Node {n}, required by boundary condition {i + 1}, is not defined"
                    )
                else:
                    nodes.append(node_map[n])
        boundary.append(
            {
                "name": name,
                "local_dof": bc["dof"],
                "type": bc["type"],
                "nodes": nodes,
                "value": bc["value"],
            }
        )

    # Convert concentrated loads to tags/vals that can be used by the assembler
    for i, cl in enumerate(inp.get("concentrated loads", [])):
        if "name" in cl:
            name = cl["name"]
        else:
            name = unique_name(inp["concentrated loads"], "CLOAD")
            cl["name"] = name
        nodes: list[int] = []
        if isinstance(cl["nodes"], str):
            if cl["nodes"] not in nsets:
                errors += 1
                logger.error(
                    f"Nodeset {cl['nodes']}, required by concentrated load {i + 1}, is not defined"
                )
            else:
                nodes.extend(nsets[cl["nodes"]])
        else:
            for n in cl["nodes"]:
                if n not in node_map:
                    errors += 1
                    logger.error(f"Node {n}, required by concentrated load {i + 1}, is not defined")
                else:
                    nodes.append(node_map[n])
        boundary.append(
            {
                "name": name,
                "local_dof": cl["dof"],
                "type": NEUMANN,
                "nodes": nodes,
                "value": cl["value"],
            }
        )

    # Process distributed load
    
    # Model length for equation-type distributed loads (assume 1D in x)
    if coords.shape[1] >= 1:
        x_coords = coords[:, 0]
        L = float(x_coords.max() - x_coords.min())
    else:
        L = 1.0

    dload: list[Any] = preprocessed.setdefault("dload", [])
    for i, dl in enumerate(inp.get("distributed loads", [])):
        if "name" in dl:
            name = dl["name"]
        else:
            name = unique_name(inp["distributed loads"], "DLOAD")
            dl["name"] = name

        # Normalize input_type; default to SCALAR for backward compatibility
        input_type = dl.get("input_type", "SCALAR")
        input_type_norm = str(input_type).upper()
        value = dl["value"]

        elems: list[int] = []
        if isinstance(dl["elements"], str):
            # elements given as set name
            if dl["elements"] not in elsets:
                errors += 1
                logger.error(
                    f"Element set {dl['elements']!r}, required by distributed load {i + 1}, not defined"
                )
            else:
                elems.extend(elsets[dl["elements"]])
        else:
            for e in dl["elements"]:
                if e not in elem_map:
                    errors += 1
                    logger.error(
                        f"Element {e}, required by distributed load {i + 1}, is not defined"
                    )
                else:
                    elems.append(elem_map[e])

        # Build q(x) callable based on input_type
        q_func = None
        if input_type_norm == "SCALAR":
            q_func = make_q_constant(value)
        elif input_type_norm == "TABLE":
            q_func = make_q_from_table(value)
        elif input_type_norm == "EQUATION":
            q_func = make_q_from_equation(value, L)

        dload.append(
            {
                "name": name,
                "elements": elems,
                "type": dl["type"],
                "direction": dl["direction"],
                "input_type": input_type_norm,
                "value": value,       # keep original for backward compatibility
                "q_func": q_func,     # normalized q(x) callable
            }
        )

    # Create a mapping from global element index to block index, local elem index (within the block)
    block_elem_map: dict[int, tuple[int, int]] = preprocessed.setdefault("block_elem_map", {})
    for ib, block in enumerate(blocks):
        for global_elem_index, local_elem_index in block["elem_map"].items():
            if global_elem_index in block_elem_map:
                errors += 1
                logger.error(f"Duplicate element ID {global_elem_index} found in multiple blocks")
            block_elem_map[global_elem_index] = (ib, local_elem_index)

    # Check if all elements are assigned to an element block
    if unassigned := set(range(num_elem)).difference(block_elem_map.keys()):
        errors += 1
        for e in unassigned:
            logger.error(f"Element {e} is not assigned to any element blocks")

    if errors:
        raise ValueError("Stopping due to previous errors")

    return preprocessed
