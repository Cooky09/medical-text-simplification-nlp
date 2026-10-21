import torch

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

from src.config import (
    PROJECT_ROOT,
    MAX_LENGTH,
    EVAL_COMPLEX_DATA_PATH,
    EVAL_SIMPLE_DATA_PATH,
)

from src.preprocessing import preprocess_text
from src.metrics import evaluate_metrics


MODEL_PATHS = {
    "bart": PROJECT_ROOT / "models" / "bart_simplification",
    "t5": PROJECT_ROOT / "models" / "t5_simplification",
    "pegasus": PROJECT_ROOT / "models" / "pegasus_simplification",
    "gpt2": PROJECT_ROOT / "models" / "gpt2_simplification",
}


def load_evaluation_data():
    """Load the evaluation complex/simple text pairs."""

    with open(
        EVAL_COMPLEX_DATA_PATH,
        "r",
        encoding="latin-1",
    ) as file:
        complex_lines = [
            line.strip()
            for line in file
            if line.strip()
        ]

    with open(
        EVAL_SIMPLE_DATA_PATH,
        "r",
        encoding="latin-1",
    ) as file:
        simple_lines = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(complex_lines) != len(simple_lines):
        raise ValueError(
            "Evaluation datasets must contain the same "
            "number of non-empty lines."
        )

    return complex_lines, simple_lines


def load_model(model_type):
    """Load one of the four models."""

    model_type = model_type.lower()

    if model_type not in MODEL_PATHS:
        raise ValueError(
            "Model must be one of: "
            "bart, t5, pegasus, gpt2"
        )

    model_path = MODEL_PATHS[model_type]

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_type.upper()} model has not been trained yet:\n"
            f"{model_path}"
        )

    if model_type == "bart":
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(
            model_path
        )

    elif model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(
            model_path
        )

    elif model_type == "pegasus":
        tokenizer = PegasusTokenizer.from_pretrained(model_path)
        model = PegasusForConditionalGeneration.from_pretrained(
            model_path
        )

    elif model_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_simplified_text(
    complex_text,
    model_type,
    model,
    tokenizer,
    device,
):
    """Generate a simplified version of medical text."""

    inputs = tokenizer(
        complex_text,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        if model_type in ["bart", "t5", "pegasus"]:

            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=MAX_LENGTH,
                num_beams=4,
                early_stopping=True,
            )

        else:
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=MAX_LENGTH,
                num_beams=4,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
            )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )


def evaluate_model(model_type):
    """
    Evaluate one model on the 10-pair evaluation dataset.
    """

    model_type = model_type.lower()

    complex_lines, simple_lines = load_evaluation_data()

    print(
        f"Evaluating {model_type.upper()} "
        f"on {len(complex_lines)} pairs..."
    )

    model, tokenizer, device = load_model(model_type)

    fkgl_scores = []
    bert_scores = []
    meteor_scores = []
    bleu_scores = []

    for complex_text, reference_text in zip(
        complex_lines,
        simple_lines,
    ):

        processed_complex = preprocess_text(
            complex_text
        )

        processed_reference = preprocess_text(
            reference_text
        )

        generated_text = generate_simplified_text(
            processed_complex,
            model_type,
            model,
            tokenizer,
            device,
        )

        scores = evaluate_metrics(
            generated_text,
            processed_reference,
        )

        fkgl_scores.append(
            scores["fkgl"]
        )

        bert_scores.append(
            scores["bertscore"]
        )

        meteor_scores.append(
            scores["meteor"]
        )

        bleu_scores.append(
            scores["bleu"]
        )

    return {
        "flesch_kincaid_grade":
            sum(fkgl_scores) / len(fkgl_scores),

        "bertscore":
            sum(bert_scores) / len(bert_scores),

        "meteor":
            sum(meteor_scores) / len(meteor_scores),

        "bleu":
            sum(bleu_scores) / len(bleu_scores),
    }


def evaluate_all_models():
    """
    Evaluate every available trained model.

    Models that have not been trained yet are skipped.
    """

    results = {}

    for model_type in MODEL_PATHS:

        try:
            results[model_type.upper()] = evaluate_model(
                model_type
            )

        except FileNotFoundError:
            print(
                f"\nSkipping {model_type.upper()}: "
                "model not trained yet."
            )

    return results


def print_results(results):
    """Print a comparison table."""

    print("\n")
    print("=" * 80)
    print("MEDICAL TEXT SIMPLIFICATION - MODEL COMPARISON")
    print("=" * 80)

    print(
        f"{'Model':<12}"
        f"{'FKGL':<12}"
        f"{'BERTScore':<14}"
        f"{'METEOR':<12}"
        f"{'BLEU':<12}"
    )

    print("-" * 80)

    for model, scores in results.items():

        print(
            f"{model:<12}"
            f"{scores['flesch_kincaid_grade']:<12.4f}"
            f"{scores['bertscore']:<14.4f}"
            f"{scores['meteor']:<12.4f}"
            f"{scores['bleu']:<12.4f}"
        )

    print("=" * 80)


if __name__ == "__main__":

    results = evaluate_all_models()

    if results:
        print_results(results)

    else:
        print(
            "\nNo trained models were found."
        )