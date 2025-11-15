import pandas as pd
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from text_reporter.helpers import clean_text, lemmatize_text
from text_reporter.utils import SPACY_MODELS, EMBEDDING_MODELS

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --------------------- CARGA DE RECURSOS -----------------------
def get_language_resources(lang: str = "multi"):
    lang = lang.lower()
    if lang not in ["en", "es", "multi"]:
        raise ValueError(f"Idioma no soportado: {lang}. Usa 'en', 'es' o 'multi'.")

    print(f"🌐 Configurando recursos para idioma: {lang.upper()}")

    # --- Stopwords
    if lang == "multi":
        sw = set(stopwords.words('spanish')) | set(stopwords.words('english'))
    else:
        sw = set(stopwords.words('english' if lang == "en" else 'spanish'))
    STOPWORDS = set(w.lower() for w in sw)

    # --- Lematización
    if lang == "en":
        lemmatizer = WordNetLemmatizer()
        nlp = None
    else:
        model_name = SPACY_MODELS[lang]
        try:
            nlp = spacy.load(model_name)
        except OSError:
            from spacy.cli import download
            print(f"Descargando modelo spaCy: {model_name}")
            download(model_name)
            nlp = spacy.load(model_name)
        lemmatizer = None

    emb_model = EMBEDDING_MODELS[lang]
    return STOPWORDS, lemmatizer, nlp, emb_model

# --------------------- PIPELINE PRINCIPAL -----------------------
def preprocess_dataframe(df: pd.DataFrame, text_column: str, lang: str = "multi") -> tuple:
    """
    Limpia y lematiza un DataFrame según idioma ('en', 'es' o 'multi').

    Returns:
        (pd.DataFrame, str): DataFrame procesado y nombre del modelo de embeddings recomendado.
    """
    STOPWORDS, lemmatizer, nlp, emb_model = get_language_resources(lang)

    processed_texts = []
    for text in df[text_column]:
        cleaned = clean_text(str(text))
        lemmatized = lemmatize_text(cleaned, lang, STOPWORDS, lemmatizer, nlp)
        processed_texts.append(lemmatized)

    df["processed_text"] = processed_texts
    df = df[df["processed_text"].str.strip().str.len() > 0].reset_index(drop=True)

    print(f"✅ {len(df)} textos procesados.")
    print(f"📦 Modelo de embeddings sugerido: {emb_model}\n")

    return df, emb_model