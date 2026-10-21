import nltk
from textstat import flesch_kincaid_grade
import evaluate


# Hugging Face evaluation metrics
bertscore_metric = evaluate.load("bertscore")
meteor_metric = evaluate.load("meteor")
bleu_metric = evaluate.load("bleu")


def evaluate_readability(text: str) -> float:
    """
    Calculate the Flesch-Kincaid Grade Level.

    Lower scores generally indicate easier-to-read text.
    """
    return flesch_kincaid_grade(text)


def evaluate_metrics(pred_text: str, ref_text: str) -> dict:
    """
    Evaluate a generated simplification against
    its reference simplification.

    Metrics:
        - Flesch-Kincaid Grade Level
        - BERTScore
        - METEOR
        - BLEU
    """

    # --------------------------------------------------
    # 1. Readability
    # --------------------------------------------------

    fkgl = evaluate_readability(pred_text)

    # --------------------------------------------------
    # 2. BERTScore
    # --------------------------------------------------

    bert_result = bertscore_metric.compute(
        predictions=[pred_text],
        references=[ref_text],
        lang="en",
    )

    # --------------------------------------------------
    # 3. METEOR
    # --------------------------------------------------

    meteor_result = meteor_metric.compute(
        predictions=[pred_text],
        references=[ref_text],
    )

    # --------------------------------------------------
    # 4. BLEU
    # --------------------------------------------------

    reference_tokens = nltk.word_tokenize(ref_text)
    prediction_tokens = nltk.word_tokenize(pred_text)

    bleu_result = bleu_metric.compute(
        predictions=[prediction_tokens],
        references=[[reference_tokens]],
    )

    return {
        "fkgl": fkgl,
        "bertscore": bert_result["f1"][0],
        "meteor": meteor_result["meteor"],
        "bleu": bleu_result["bleu"],
    }