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

from src.preprocessing import preprocess_text
from src.config import MAX_LENGTH


def load_model(model_type, model_path):
    """
    Load one of the four trained simplification models.

    Supported:
        bart
        t5
        pegasus
        gpt2
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_type = model_type.lower()

    if model_type == "bart":
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)

    elif model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)

    elif model_type == "pegasus":
        tokenizer = PegasusTokenizer.from_pretrained(model_path)
        model = PegasusForConditionalGeneration.from_pretrained(model_path)

    elif model_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)

        model = GPT2LMHeadModel.from_pretrained(model_path)

        # GPT-2 does not have a padding token by default.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    else:
        raise ValueError(
            "Unsupported model. Choose: bart, t5, pegasus, or gpt2."
        )

    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_simplification(
    text,
    model_type,
    model_path,
):
    """
    Generate a simplified version of medical text.
    """

    model, tokenizer, device = load_model(
        model_type,
        model_path,
    )

    processed_text = preprocess_text(text)

    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        if model_type.lower() in [
            "bart",
            "t5",
            "pegasus",
        ]:

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

    result = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return result


if __name__ == "__main__":

    text = input("\nEnter medical text:\n")

    print("\nChoose a model:")
    print("1. BART")
    print("2. T5")
    print("3. PEGASUS")
    print("4. GPT-2")

    choice = input("\nChoice: ").strip()

    model_map = {
        "1": "bart",
        "2": "t5",
        "3": "pegasus",
        "4": "gpt2",
    }

    if choice not in model_map:
        raise ValueError("Invalid model choice.")

    model_type = model_map[choice]

    print(f"\nSelected model: {model_type.upper()}")
    print("\nSimplified text:\n")

    # Model path will be configured later.
    print(
        "Model path has not been configured yet. "
        "Train or provide the selected model first."
    )