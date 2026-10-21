import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Clean and lemmatize text using the preprocessing
    used in the original notebook.
    """
    tokens = word_tokenize(text.lower())
    processed = []

    for word, tag in pos_tag(tokens):
        if word.isalnum() and word not in stop_words:
            pos = (
                "a" if tag.startswith("J")
                else "v" if tag.startswith("V")
                else "n" if tag.startswith("N")
                else "r" if tag.startswith("R")
                else "n"
            )
            processed.append(lemmatizer.lemmatize(word, pos))

    return " ".join(processed)