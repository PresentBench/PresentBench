from pathlib import Path


def find_material_files(base_dir: Path) -> list[Path]:
    """
    Find all material files.
    Supported formats:
    - material.md, material.pdf
    - material_1.md, material_1.pdf, material_2.md, material_2.pdf, ...
    
    Returns a list sorted by index (material first, then material_1, material_2, ...).
    """
    material_files = []
    base_dir = Path(base_dir)
    
    # First check material.md and material.pdf (no index).
    for ext in ['.md', '.pdf']:
        material_file = base_dir / f"material{ext}"
        if material_file.exists():
            material_files.append(material_file)
            break  # Only take one unindexed file (prefer .md).
    
    # Then check indexed material files (starting from 1, consecutive positive integers).
    i = 1
    while True:
        found_any = False
        for ext in ['.md', '.pdf']:
            material_file = base_dir / f"material_{i}{ext}"
            if material_file.exists():
                material_files.append(material_file)
                found_any = True
                break  # Only take one per index (prefer .md).
        
        if not found_any:
            break  # Missing this index means we've reached the end.
        i += 1
    
    return material_files
