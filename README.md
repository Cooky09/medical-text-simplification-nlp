# Text Simplification using Transformer Models (BART, T5, PEGASUS)

## Overview
This project provides a **research-grade pipeline** for simplifying complex English text into simpler, more readable sentences using modern transformer-based sequence-to-sequence models.  
It is designed to be clean, modular, and reproducible, making it suitable for academic research, thesis work, or production experimentation.

Supported models:
- **BART** (`facebook/bart-base`)
- **T5** (`t5-small`)
- **PEGASUS** (`google/pegasus-xsum`)

The project performs:
- Text preprocessing (tokenization, stopword removal, lemmatization)
- Dataset loading of aligned complex-simple text
- Model training with configurable hyperparameters
- Evaluation using BLEU, METEOR, BERTScore, and readability metrics
- Saving results with original, target, and generated texts

---

## Project Structure
