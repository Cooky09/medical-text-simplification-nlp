from pathlib import Path

import pandas as pd


def load_parallel_text(
    complex_path: str | Path,
    simple_path: str | Path,
) -> pd.DataFrame:
    """
    Load aligned complex and simplified medical text.

    Each non-empty line in complex_path must correspond to the
    same non-empty line in simple_path.
    """

    complex_path = Path(complex_path)
    simple_path = Path(simple_path)

    with complex_path.open("r", encoding="latin-1") as file:
        complex_lines = [
            line.strip()
            for line in file
            if line.strip()
        ]

    with simple_path.open("r", encoding="latin-1") as file:
        simple_lines = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(complex_lines) != len(simple_lines):
        raise ValueError(
            "Complex and simple datasets must contain the same "
            f"number of non-empty lines. "
            f"Found {len(complex_lines)} complex lines and "
            f"{len(simple_lines)} simple lines."
        )

    return pd.DataFrame(
        {
            "Original": complex_lines,
            "Simple": simple_lines,
        }
    )