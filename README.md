# Medical Text Simplification using NLP

A Natural Language Processing project for simplifying complex medical text into easier-to-understand language using transformer-based deep learning models.

The project compares four different language models for medical text simplification:

- **BART**
- **T5**
- **PEGASUS**
- **GPT-2**

The models are trained and evaluated on paired medical text containing complex medical descriptions and their simplified versions.

---

## Project Overview

Medical documents often contain technical terminology and complex sentence structures that can be difficult for patients and non-specialists to understand.

This project investigates whether NLP and transformer-based language models can automatically simplify medical text while preserving the important medical information.

### Example

**Original medical text:**

> The patient presented with an acute exacerbation of chronic obstructive pulmonary disease characterized by increased dyspnea, wheezing, and productive cough despite previous bronchodilator and corticosteroid therapy.

**BART simplified output:**

> patient come acute exacerbation chronic obstructive pulmonary disease characterize increased wheezing cough despite use bronchodilator corticosteroid therapy

The system removes some complex grammatical structures while retaining important medical concepts.

---

## Models

Four transformer-based models were trained and compared.

| Model | Purpose |
|---|---|
| BART | Sequence-to-sequence text simplification |
| T5 | Text-to-text transformation |
| PEGASUS | Abstractive text generation |
| GPT-2 | Autoregressive text generation |

All models were trained using the same medical text simplification dataset and evaluated using the same evaluation dataset.

---

## Model Performance

The current evaluation results are:

| Model | FKGL ↓ | BERTScore ↑ | METEOR ↑ | BLEU ↑ |
|---|---:|---:|---:|---:|
| **BART** | **21.5348** | **0.9604** | **0.8041** | **0.6229** |
| T5 | 31.5811 | 0.8792 | 0.3556 | 0.0604 |
| PEGASUS | 28.2393 | 0.8703 | 0.3117 | 0.0476 |
| GPT-2 | 33.5696 | 0.8774 | 0.3668 | 0.0474 |

### Best performing model

Based on the current evaluation:

**BART performed substantially better than the other tested models.**

It achieved:

- Lowest FKGL score among the tested models
- Highest BERTScore
- Highest METEOR
- Highest BLEU

Therefore, BART is currently the strongest model in this project.

> Note: These results are based on a relatively small evaluation dataset and should not be interpreted as clinical validation.

---

## Dataset

The project uses parallel medical text data.

Each training example contains:

```text
Complex medical text
        ↓
Simplified medical text
```

The dataset was validated before training.

### Dataset validation

Training dataset:

- Complex records: **269**
- Simple records: **269**
- Aligned pairs: **True**
- Empty complex records: **0**
- Empty simple records: **0**

Evaluation dataset:

- Complex records: **10**
- Simple records: **10**
- Aligned pairs: **True**
- Empty complex records: **0**
- Empty simple records: **0**

Dataset validation status:

**PASS**

---

## Project Structure

```text
medical-text-simplification-nlp/
│
├── data/
│   ├── complex_text.txt
│   ├── original_text.txt
│   ├── simple_text.txt
│   └── simplified_text.txt
│
├── notebooks/
│   ├── Medical_Text_Simplification_GPT2.ipynb
│   ├── Medical_Text_simplification (1).ipynb
│   ├── PEGASUS_Text_simplification (1).ipynb
│   └── T5_Text_simplification.ipynb
│
├── results/
│   ├── T5_evaluation.xlsx
│   └── bart_evaluation.xlsx
│
├── src/
│   ├── compare_models.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── test_dataset.py
│   ├── train_bart.py
│   ├── train_gpt2.py
│   ├── train_pegasus.py
│   └── train_t5.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Cooky09/medical-text-simplification-nlp.git
```

Move into the project directory:

```bash
cd medical-text-simplification-nlp
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset Validation

Before training, the dataset can be validated using:

```bash
python -m src.test_dataset
```

The validation checks:

- Number of complex records
- Number of simplified records
- Alignment between the datasets
- Empty records
- Dataset consistency

---

## Training

Each model has its own training script.

### BART

```bash
python -m src.train_bart
```

### T5

```bash
python -m src.train_t5
```

### PEGASUS

```bash
python -m src.train_pegasus
```

### GPT-2

```bash
python -m src.train_gpt2
```

The trained models are saved locally under the configured model output directories.

---

## Model Configuration

Training parameters are controlled through:

```text
src/config.py
```

The project uses parameters such as:

```text
MAX_LENGTH = 1024
BATCH_SIZE = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS = 15
```

These values can be modified depending on the available hardware and dataset.

---

## Evaluation

After training the models, the evaluation script can be used to compare their performance:

```bash
python -m src.evaluate
```

The evaluation uses:

- **Flesch-Kincaid Grade Level (FKGL)**
- **BERTScore**
- **METEOR**
- **BLEU**

### Metric interpretation

#### FKGL

Measures the approximate reading grade level required to understand the generated text.

Lower values generally indicate easier-to-read text.

#### BERTScore

Measures semantic similarity between the generated text and reference text using contextual embeddings.

Higher values indicate greater semantic similarity.

#### METEOR

Measures similarity using factors such as word matching and word order.

Higher values are generally better.

#### BLEU

Measures overlap between generated text and reference text using n-gram precision.

Higher values indicate greater overlap.

---

## Comparing Models

The project also includes a model comparison script:

```bash
python -m src.compare_models
```

This can be used to compare the trained models using the available evaluation metrics.

---

## Text Prediction

The prediction system allows a user to enter medical text and select a trained model.

Run:

```bash
python -m src.predict
```

The system provides a menu:

```text
Choose a model:

1. BART
2. T5
3. PEGASUS
4. GPT-2
```

The selected model then generates a simplified version of the medical text.

---

## Example Prediction

### Input

```text
The patient with type 2 diabetes mellitus presented with increased thirst, frequent urination, and elevated blood glucose levels.
```

### BART output

```text
patient type 2 diabetes mellitus experience increased thirst frequent urination raise blood glucose level
```

Another example:

### Input

```text
The patient was diagnosed with hypertension after repeated measurements showed persistently elevated blood pressure despite lifestyle modifications.
```

### BART output

```text
patient diagnose hypertension repeat test show high blood pressure despite lifestyle change
```

---

## Hardware

Training transformer models requires significant computational resources.

A GPU was used during the training process.

The project was tested using a Google Colab environment with a:

```text
NVIDIA Tesla T4 GPU
```

Training time and memory requirements may vary depending on:

- GPU
- batch size
- sequence length
- model architecture
- number of training epochs
- dataset size

---

## Trained Models

The trained model weights are intentionally **not stored in this GitHub repository** because they are large files.

The repository contains the code required to:

1. Download the base transformer models.
2. Prepare the dataset.
3. Train the models.
4. Evaluate the models.
5. Run predictions.

Large model files such as:

```text
*.safetensors
*.bin
*.pt
*.pth
```

are excluded through `.gitignore`.

The trained model backups are maintained separately.

---

## Reproducing the Results

To reproduce the experiments:

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Validate the dataset

```bash
python -m src.test_dataset
```

### 3. Train the models

```bash
python -m src.train_bart
python -m src.train_t5
python -m src.train_pegasus
python -m src.train_gpt2
```

### 4. Evaluate the models

```bash
python -m src.evaluate
```

### 5. Compare the models

```bash
python -m src.compare_models
```

### 6. Run interactive prediction

```bash
python -m src.predict
```

---

## Limitations

This project is an NLP research and educational project and has several limitations.

### Dataset size

The training dataset contains only **269 aligned training pairs**, with **10 evaluation pairs**.

This is relatively small for training large transformer models.

### Evaluation size

The evaluation dataset contains only 10 examples, so the reported metrics may not generalize to larger medical datasets.

### Medical accuracy

Text simplification can accidentally remove, change, or introduce important medical information.

The generated text should therefore **not be used as a substitute for professional medical advice or clinical decision-making**.

### Grammar

The preprocessing pipeline intentionally simplifies the language, and some generated outputs may not be grammatically natural.

### Model comparison

The four models have different architectures and generation characteristics. The reported scores should therefore be interpreted as experimental results rather than definitive evidence that one architecture is universally better.

---

## Future Work

Possible improvements include:

- Increasing the size of the medical training dataset.
- Using a larger and more diverse evaluation dataset.
- Improving preprocessing.
- Improving grammatical quality.
- Adding medical terminology preservation checks.
- Adding factual consistency evaluation.
- Performing human evaluation with medical professionals.
- Fine-tuning hyperparameters.
- Using early stopping and validation-based model selection.
- Testing additional transformer architectures.
- Adding a web-based user interface.
- Deploying the best-performing model as an API.
- Adding confidence and safety checks for generated text.

---

## Technologies

The project uses:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Tokenizers
- NLTK
- BERTScore
- METEOR
- BLEU
- TextStat
- pandas
- NumPy
- Google Colab

---

## Disclaimer

This project is intended for **research, education, and experimentation in Natural Language Processing**.

The generated simplified medical text may contain errors, omissions, or inaccurate statements.

**It must not be used for diagnosis, treatment decisions, or other clinical decision-making.**

Always consult a qualified healthcare professional for medical advice.

---

## Author

**Cooky09**

GitHub:

https://github.com/Cooky09

Project repository:

https://github.com/Cooky09/medical-text-simplification-nlp

---

## License

This project is provided for educational and research purposes.

Please check the licenses of the underlying pretrained models and datasets before using this project for commercial purposes.
