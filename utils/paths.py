from os import PathLike
from pathlib import Path


"""
Path notes
- All string paths in this file are relative to the data root (data/), not to the current working directory (cwd).
- This ensures paths resolve consistently no matter where you run the code from.

To switch to another dataset with the same directory structure:
- Option 1: call get_data_source_dirs(base_dir=Path("/your/other/data"))
"""


DATA_SOURCE_DIRS_REL = [
    # academia
    'academia/CVPR_2023',
    'academia/CVPR_2024',
    'academia/CVPR_2025',
    'academia/F1000_talks',
    'academia/ICLR_2024',
    'academia/ICLR_2025',
    'academia/ICML_2024',
    'academia/ICML_2025',
    'academia/NeurIPS_2024',
    'academia/FAST_2025',
    'academia/NSDI_2025',
    'academia/OSDI_2025',
    'academia/USENIX',
    'academia/NBER_conferences',
    
    # advertising
    'advertising/Apple_iPad',
    'advertising/Apple_iPhone',
    'advertising/Apple_Mac',
    'advertising/BMW',
    'advertising/lenovo',
    
    # education
    'education/Computer_science_lectures',
    'education/CSAPP-Lectures_2015Fall',
    'education/MIT-Financing_Economic_Development',
    'education/MIT-the_human_brain',
    'education/THU_DSA',
    
    # economics
    'economics/Alphabet_Investor_Relations',
    'economics/JPMorgan_Chase',
    'economics/Microsoft_press_release',
    'economics/OECD_Economic_Outlook',
    'economics/TESLA_update_letter',
    'economics/World_Bank_GPE',
    
    # talk
    'talk/middle_school_presentation',
    'talk/ted_chinese',
    'talk/US_presidential_speech',
]

DATA_SOURCE_DIRS_REL = [Path(rel) for rel in DATA_SOURCE_DIRS_REL]

SKIP_NAMES = {"__pycache__"}

def get_data_source_dirs(base_dir: PathLike | None = None) -> list[Path]:
    """
    Return data source directories (absolute paths).

    - When base_dir is None:
      * If `data/` exists under the repository root, use it as the data root.
      * Otherwise fall back to the repo root itself (for legacy directory layouts).
    - When base_dir is provided, switch to another data root with the same structure.
    """
    if base_dir is None:
        repo_root = Path(__file__).resolve().parents[1]
        root = repo_root / "data"
    else:
        root = Path(base_dir)
    root = root.expanduser().resolve()
    return [root / rel for rel in DATA_SOURCE_DIRS_REL]



def get_valid_item_dirs(src_dir: Path, skip_names: set[str] | None = None) -> list[Path]:
    """
    Get valid subdirectories under src_dir.

    Excludes:
    - Non-directories
    - Names in skip_names (defaults to SKIP_NAMES)
    - Hidden directories starting with '.'
    
    Args:
        src_dir: Source directory path.
        skip_names: Directory names to skip (defaults to SKIP_NAMES).
        
    Returns:
        List of valid subdirectory paths.
        
    Raises:
        OSError: If the directory cannot be read.
    """
    if skip_names is None:
        skip_names = set()
    
    return [
        p
        for p in src_dir.iterdir()
        if p.is_dir() and p.name not in skip_names and not p.name.startswith(".")
    ]


def get_all_data_item_dirs(base_dir: PathLike | None = None, skip_names: set[str] | None = None, return_rel_path: bool = False) -> list[Path]:
    """
    Combine get_data_source_dirs and get_valid_item_dirs.
    
    Args:
        base_dir: Data root directory; if None, use the default data root.
        skip_names: Directory names to skip (defaults to SKIP_NAMES).
        
    Returns:
        Flattened list of valid item directories under all data source dirs.
    """
    all_dirs: list[Path] = []
    for src_dir in get_data_source_dirs(base_dir=base_dir):
        all_dirs.extend(get_valid_item_dirs(src_dir, skip_names=skip_names))
    
    if return_rel_path:
        all_dirs = [dir.relative_to(base_dir) for dir in all_dirs]
    
    return all_dirs

