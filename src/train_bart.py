import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration

from src.config import (
    COMPLEX_DATA_PATH,
    SIMPLE_DATA_PATH,
    BART_MODEL_NAME,
    MAX_LENGTH,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MODEL_OUTPUT_DIR,
)

from src.dataset import load_parallel_text
from src.preprocessing import preprocess_text


def prepare_dataset(df):
    """
    Preprocess complex and simplified medical text.
    """
    df = df.copy()

    df["input_text"] = df["Original"].apply(preprocess_text)
    df["target_text"] = df["Simple"].apply(preprocess_text)

    return df


def tokenize_dataset(df, tokenizer):
    """
    Tokenize input and target text for BART.

    Padding tokens in the target are replaced with -100
    so they are ignored when calculating the loss.
    """

    inputs = tokenizer(
        df["input_text"].tolist(),
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )

    targets = tokenizer(
        df["target_text"].tolist(),
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )

    dataset = []

    for i in range(len(df)):

        label_ids = targets["input_ids"][i]

        # Ignore padding tokens during loss calculation.
        label_ids = [
            token if token != tokenizer.pad_token_id else -100
            for token in label_ids
        ]

        dataset.append(
            {
                "input_ids": torch.tensor(
                    inputs["input_ids"][i],
                    dtype=torch.long,
                ),
                "attention_mask": torch.tensor(
                    inputs["attention_mask"][i],
                    dtype=torch.long,
                ),
                "labels": torch.tensor(
                    label_ids,
                    dtype=torch.long,
                ),
            }
        )

    return dataset


def create_training_components():
    """
    Load data and create all BART training components.
    """

    df = load_parallel_text(
        COMPLEX_DATA_PATH,
        SIMPLE_DATA_PATH,
    )

    print(f"Loaded {len(df)} aligned pairs.")

    df = prepare_dataset(df)

    print("\nExample after preprocessing:")
    print("INPUT :", df.iloc[0]["input_text"][:300])
    print("TARGET:", df.iloc[0]["target_text"][:300])

    tokenizer = BartTokenizer.from_pretrained(
        BART_MODEL_NAME
    )

    model = BartForConditionalGeneration.from_pretrained(
        BART_MODEL_NAME
    )

    tokenized_dataset = tokenize_dataset(
        df,
        tokenizer,
    )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
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
    Train BART for medical text simplification.
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

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

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

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        MODEL_OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        MODEL_OUTPUT_DIR
    )

    print(
        f"\nModel saved to: {MODEL_OUTPUT_DIR}"
    )

    return losses


if __name__ == "__main__":
    train()