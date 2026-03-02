# Text Simplification using Transformer Models (BART, T5, PEGASUS)

📌 Project Overview
Medical jargon and complex clinical terminology often create a barrier between healthcare providers and patients. This project leverages Natural Language Processing (NLP) to automatically simplify technical medical texts into "layman’s terms" without losing the original clinical meaning.

The goal is to improve health literacy by transforming professional medical abstracts (from sources like PubMed or Cochrane) into easy-to-read sentences for patients.

🚀 Features
Technical-to-Simple Translation: Converts complex medical sentences into plain English.

Jargon Identification: (If applicable) Highlights difficult terms and replaces them with synonyms.

Evaluation Pipeline: Includes scripts to calculate SARI, BLEU, and ROUGE scores to measure simplification quality.

Fine-tuned Models: Implementation of state-of-the-art architectures like T5-Base, BART-Large, or BioGPT.

📂 Project Structure

├── data/                   # Raw and preprocessed datasets
├── models/                 # Saved model checkpoints and weights
├── notebooks/              # Jupyter notebooks for EDA and testing
├── src/
│   ├── preprocess.py       # Data cleaning and tokenization
│   ├── train.py            # Training and fine-tuning scripts
│   └── evaluate.py         # Metrics calculation (SARI, BLEU, etc.)
├── requirements.txt        # Project dependencies
└── README.md


🛠️ Installation
Clone the repository:

Bash
git clone https://github.com/Cooky09/medical-text-simplification-nlp.git
cd medical-text-simplification-nlp
Create a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
📊 Dataset
This project uses the following datasets for training and validation:

[Dataset Name, e.g., MedSiML or Cochrane]: Contains over XX,XXX pairs of original vs. simplified medical sentences.

Preprocessing: We apply lowercasing, removal of special characters, and length filtering to ensure high-quality training pairs.

🧪 Usage
1. Training the Model
To start fine-tuning the model on your dataset:

Bash
python src/train.py --model_name "t5-small" --epochs 3 --batch_size 8
2. Inference (Simplifying Text)
Use the following script to simplify a custom sentence:

Python
from src.predict import simplifier

text = "The patient presents with acute myocardial infarction and secondary tachycardia."
result = simplifier.simplify(text)
print(result) # Output: The patient is having a heart attack and a fast heartbeat.
