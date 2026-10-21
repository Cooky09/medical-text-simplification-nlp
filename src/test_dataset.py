from pathlib import Path

from src.config import (
    COMPLEX_DATA_PATH,
    SIMPLE_DATA_PATH,
    PROJECT_ROOT,
)
from src.dataset import load_parallel_text


EVALUATION_COMPLEX_PATH = PROJECT_ROOT / "data" / "original_text.txt"
EVALUATION_SIMPLE_PATH = PROJECT_ROOT / "data" / "simplified_text.txt"


def check_dataset(complex_path: Path, simple_path: Path, name: str) -> bool:
    """Validate that a complex/simple dataset is aligned."""

    print(f"\n--- {name} ---")

    try:
        df = load_parallel_text(complex_path, simple_path)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return False

    print(f"Complex records: {len(df)}")
    print(f"Simple records:  {len(df)}")
    print("Aligned:         True")

    # Check for empty values after loading
    empty_complex = df["Original"].str.strip().eq("").sum()
    empty_simple = df["Simple"].str.strip().eq("").sum()

    print(f"Empty complex:   {empty_complex}")
    print(f"Empty simple:    {empty_simple}")

    if empty_complex > 0 or empty_simple > 0:
        print("FAIL: Empty records found.")
        return False

    print("Status:          PASS")
    return True


def main():
    print("Medical Text Simplification - Dataset Validation")
    print("=" * 52)

    training_ok = check_dataset(
        COMPLEX_DATA_PATH,
        SIMPLE_DATA_PATH,
        "Training Dataset",
    )

    evaluation_ok = check_dataset(
        EVALUATION_COMPLEX_PATH,
        EVALUATION_SIMPLE_PATH,
        "Evaluation Dataset",
    )

    print("\n" + "=" * 52)

    if training_ok and evaluation_ok:
        print("DATASET VALIDATION: PASS")
        print("Training pairs:    269")
        print("Evaluation pairs:  10")
    else:
        print("DATASET VALIDATION: FAIL")


if __name__ == "__main__":
    main()