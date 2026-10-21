import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from src.config import (
    COMPLEX_DATA_PATH,
    SIMPLE_DATA_PATH,
    MAX_LENGTH,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PROJECT_ROOT,
)

from src.dataset import load_parallel_text
from src.preprocessing import preprocess_text


GPT2_MODEL_NAME = "gpt2"

GPT2_OUTPUT_DIR = (
    PROJECT_ROOT / "models" / "gpt2_simplification"
)


def prepare_dataset(df):
    """
    Preprocess complex and simplified medical text.
    """

    df = df.copy()

    df["input_text"] = df["Original"].apply(
        preprocess_text
    )

    df["target_text"] = df["Simple"].apply(
        preprocess_text
    )

    return df


def build_gpt2_dataset(df, tokenizer):
    """
    Create causal-language-model training examples.

    GPT-2 receives:

        simplify medical text: <complex text>
        simplified: <simple text>

    The loss is calculated only on the simplified target
    and not on the prompt.
    """

    dataset = []

    for _, row in df.iterrows():

        prompt = (
            "simplify medical text: "
            + row["input_text"]
            + "\n"
            + "simplified: "
        )

        target = row["target_text"]

        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        target_ids = tokenizer.encode(
            target,
            add_special_tokens=False,
        )

        # Add EOS to the target so GPT-2 learns where
        # the generated simplification should end.
        target_ids = target_ids + [
            tokenizer.eos_token_id
        ]

        full_ids = prompt_ids + target_ids

        # Truncate to the maximum sequence length.
        full_ids = full_ids[:MAX_LENGTH]

        input_ids = full_ids.copy()

        attention_mask = [
            1 for _ in input_ids
        ]

        # Ignore the prompt when calculating loss.
        labels = (
            [-100] * len(prompt_ids)
            + target_ids
        )

        # Keep labels aligned with the truncated sequence.
        labels = labels[:MAX_LENGTH]

        # Padding.
        padding_length = (
            MAX_LENGTH - len(input_ids)
        )

        input_ids += (
            [tokenizer.pad_token_id]
            * padding_length
        )

        attention_mask += (
            [0] * padding_length
        )

        labels += (
            [-100] * padding_length
        )

        dataset.append(
            {
                "input_ids": torch.tensor(
                    input_ids,
                    dtype=torch.long,
                ),
                "attention_mask": torch.tensor(
                    attention_mask,
                    dtype=torch.long,
                ),
                "labels": torch.tensor(
                    labels,
                    dtype=torch.long,
                ),
            }
        )

    return dataset


def create_training_components():
    """
    Load data and create all GPT-2 training components.
    """

    df = load_parallel_text(
        COMPLEX_DATA_PATH,
        SIMPLE_DATA_PATH,
    )

    print(
        f"Loaded {len(df)} aligned pairs."
    )

    df = prepare_dataset(df)

    print("\nExample after preprocessing:")

    print(
        "INPUT :",
        df.iloc[0]["input_text"][:300],
    )

    print(
        "TARGET:",
        df.iloc[0]["target_text"][:300],
    )

    tokenizer = GPT2Tokenizer.from_pretrained(
        GPT2_MODEL_NAME
    )

    # GPT-2 does not have a default padding token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(
        GPT2_MODEL_NAME
    )

    # Make sure model configuration agrees with tokenizer.
    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    tokenized_dataset = build_gpt2_dataset(
        df,
        tokenizer,
    )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.95,
    )

    return (
        df,
        tokenizer,
        model,
        dataloader,
        device,
        optimizer,
        scheduler,
    )


def train():
    """
    Train GPT-2 for medical text simplification.
    """

    (
        df,
        tokenizer,
        model,
        dataloader,
        device,
        optimizer,
        scheduler,
    ) = create_training_components()

    losses = []

    for epoch in range(NUM_EPOCHS):

        model.train()

        total_loss = 0.0

        for batch in dataloader:

            optimizer.zero_grad()

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            total_loss += loss.item()

            loss.backward()

            optimizer.step()

        scheduler.step()

        average_loss = (
            total_loss / len(dataloader)
        )

        losses.append(average_loss)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"- Loss: {average_loss:.4f}"
        )

    GPT2_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        GPT2_OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        GPT2_OUTPUT_DIR
    )

    print(
        f"\nModel saved to: "
        f"{GPT2_OUTPUT_DIR}"
    )

    return losses


if __name__ == "__main__":
    train()