# Medical Text Simplification NLP

Automatically simplifies complex medical text into easier-to-understand language using modern NLP models.

Medical jargon and clinical terminology can be difficult for patients and non-experts to understand. This project builds a modular pipeline to fine-tune and evaluate transformer models (BART, T5, PEGASUS) for medical text simplification with reliable evaluation metrics.

---

## 🚀 Features

- **Preprocessing:** Tokenization, stop word removal, and lemmatization for clean input.
- **Flexible Dataset Loader:** Loads parallel complex–simple text pairs.
- **Multi-Model Training:** Train BART, T5, or PEGASUS models with a unified training script.
- **Evaluation Metrics:** BLEU, METEOR, BERTScore, and Flesch Reading Ease.
- **Detailed Results:** CSV outputs include original text, target simplified text, and generated simplifications.
- **Research-Ready:** Clean architecture with reproducible experiments and metric tracking.

---

## 📁 Project Structure


.
├── data/ # Complex and simplified text files
├── models/ # Saved model checkpoints
├── results/ # Evaluation results CSVs
├── src/
│ ├── preprocessing.py # Text cleaning and lemmatization
│ ├── dataset.py # Dataset loading
│ ├── config.py # Model & hyperparameter config
│ ├── train.py # Multi-model training script
│ ├── metrics.py # Evaluation metric functions
│ └── evaluate.py # Model evaluation & CSV export
├── notebooks/ # Optional EDA & testing notebooks
└── requirements.txt # Python dependencies


---

## 🧠 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/Cooky09/medical-text-simplification-nlp.git
cd medical-text-simplification-nlp
pip install -r requirements.txt

Make sure you have PyTorch, Transformers, and NLTK installed.

📌 Prepare Your Dataset

Place aligned parallel text in:

data/complex_text.txt

data/simple_text.txt

Each line in the simple file must correspond to the same line in the complex file.

🎯 Training

Train any supported model with:

cd src
python train.py --model bart
python train.py --model t5
python train.py --model pegasus

Training parameters are controlled via src/config.py.

📊 Evaluation

Evaluate a trained model:

python evaluate.py --model bart

Evaluation produces a CSV in results/ with:

| original_text | target_text | generated_text | BLEU | METEOR | BERTScore_F1 | Readability |

This makes comparisons easy and suitable for research reporting.

📈 Supported Models
Model	Description
BART	Encoder–decoder model good for text generation
T5	Text-to-Text model for sequence tasks
PEGASUS	Optimized for summarization and simplification
📚 Evaluation Metrics Explained

BLEU: Measures overlap between generated and reference text.

METEOR: Accounts for synonyms and paraphrasing.

BERTScore: Semantic similarity using contextual embeddings.

Readability: Flesch Reading Ease for simple text readability.

🧪 Example Workflow

Prepare your dataset.

Train your model (e.g., BART).

Evaluate and review output CSV in results/.

📝 Notes

This repository is designed for research and experimentation in medical NLP, focusing on accessibility and health literacy by transforming technical text into layman-friendly language.
