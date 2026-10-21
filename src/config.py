from pathlib import Path


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# Dataset paths
# ============================================================

COMPLEX_DATA_PATH = PROJECT_ROOT / "data" / "complex_text.txt"
SIMPLE_DATA_PATH = PROJECT_ROOT / "data" / "simple_text.txt"

EVAL_COMPLEX_DATA_PATH = PROJECT_ROOT / "data" / "original_text.txt"
EVAL_SIMPLE_DATA_PATH = PROJECT_ROOT / "data" / "simplified_text.txt"


# ============================================================
# Model names
# ============================================================

BART_MODEL_NAME = "facebook/bart-large"

T5_MODEL_NAME = "t5-base"

PEGASUS_MODEL_NAME = "google/pegasus-xsum"

GPT2_MODEL_NAME = "gpt2"


# ============================================================
# Tokenization
# ============================================================

MAX_LENGTH = 1024


# ============================================================
# Training
# ============================================================

BATCH_SIZE = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS = 75


# ============================================================
# Model output directories
# ============================================================

BART_OUTPUT_DIR = (
    PROJECT_ROOT / "models" / "bart_simplification"
)

T5_OUTPUT_DIR = (
    PROJECT_ROOT / "models" / "t5_simplification"
)

PEGASUS_OUTPUT_DIR = (
    PROJECT_ROOT / "models" / "pegasus_simplification"
)

GPT2_OUTPUT_DIR = (
    PROJECT_ROOT / "models" / "gpt2_simplification"
)