from pathlib import Path

from src.config import PROJECT_ROOT


# Model output directories
MODEL_PATHS = {
    "BART": PROJECT_ROOT / "models" / "bart_simplification",
    "T5": PROJECT_ROOT / "models" / "t5_simplification",
    "PEGASUS": PROJECT_ROOT / "models" / "pegasus_simplification",
    "GPT-2": PROJECT_ROOT / "models" / "gpt2_simplification",
}


def check_model_paths():
    """
    Check whether trained model directories exist.
    """

    print("Medical Text Simplification - Model Check")
    print("=" * 50)

    available_models = []

    for model_name, model_path in MODEL_PATHS.items():

        exists = model_path.exists()

        status = "AVAILABLE" if exists else "NOT TRAINED"

        print(f"{model_name:<10} : {status}")
        print(f"  Path: {model_path}")

        if exists:
            available_models.append(model_name)

    print("=" * 50)

    print(
        f"Available models: "
        f"{len(available_models)}/{len(MODEL_PATHS)}"
    )

    return available_models


def create_comparison_template():
    """
    Create an empty comparison structure for the four models.

    Actual metric values will be filled after evaluation.
    """

    results = {
        "BART": {
            "flesch_kincaid_grade": None,
            "bertscore": None,
            "meteor": None,
            "bleu": None,
        },
        "T5": {
            "flesch_kincaid_grade": None,
            "bertscore": None,
            "meteor": None,
            "bleu": None,
        },
        "PEGASUS": {
            "flesch_kincaid_grade": None,
            "bertscore": None,
            "meteor": None,
            "bleu": None,
        },
        "GPT-2": {
            "flesch_kincaid_grade": None,
            "bertscore": None,
            "meteor": None,
            "bleu": None,
        },
    }

    return results


def print_comparison_template():
    """
    Display the planned four-model evaluation table.
    """

    results = create_comparison_template()

    print("\nModel Comparison")
    print("=" * 80)

    print(
        f"{'Model':<12}"
        f"{'FKGL':<12}"
        f"{'BERTScore':<14}"
        f"{'METEOR':<12}"
        f"{'BLEU':<12}"
    )

    print("-" * 80)

    for model_name, scores in results.items():

        print(
            f"{model_name:<12}"
            f"{str(scores['flesch_kincaid_grade']):<12}"
            f"{str(scores['bertscore']):<14}"
            f"{str(scores['meteor']):<12}"
            f"{str(scores['bleu']):<12}"
        )


if __name__ == "__main__":
    check_model_paths()
    print_comparison_template()