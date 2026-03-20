"""
Common utility: generate prefixed checklists.
"""
from __future__ import annotations
from typing import Any, Sequence

from utils.count_pages import count_pages


# Add prefix to each item (if it is a str).
def add_prefix_to_strings(lst: Sequence[Any], prefix: str) -> list[Any]:
    return [prefix + item if isinstance(item, str) else item for item in lst]


def generate_checklist(
    slides_path: str,
    material_independent_checklist: list[list[Any]],
    material_dependent_checklist: list[list[Any]],
    material_independent_prefix: str,
    material_dependent_prefix: str,
) -> tuple[list[list[Any]], list[list[Any]]]:
    if len(material_independent_checklist) != 2:
        raise ValueError("material_independent_checklist must contain 2 elements")
    if len(material_dependent_checklist) != 3:
        raise ValueError("material_dependent_checklist must contain 3 elements")

    material_independent_checklist = [
        add_prefix_to_strings(checklist, material_independent_prefix)
        for checklist in material_independent_checklist
    ]
    material_dependent_checklist = [
        add_prefix_to_strings(checklist, material_dependent_prefix)
        for checklist in material_dependent_checklist
    ]
    material_dependent_checklist[2] = material_dependent_checklist[2][:count_pages(slides_path)]

    return material_independent_checklist, material_dependent_checklist
