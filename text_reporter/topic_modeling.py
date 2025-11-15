import numpy as np
import umap
import pickle
import os

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

def train_topic_model(
    df,
    text_column="processed_text",
    embeddings_path="data/embeddings.npy",
    language="en"
):
    """
    Entrena un modelo BERTopic replicando la configuración anterior (menos tópicos).
    """

    print(f"\nCargando embeddings desde: {embeddings_path}")
    embeddings = np.load(embeddings_path)

    # --- UMAP 
    print("Reduciendo dimensionalidad con UMAP...")
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=10,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    # --- HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=25,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    # --- Vectorizer (usa ngramas)
    stop_words_param = "english" if language == "en" else None
    vectorizer_model = CountVectorizer(stop_words=stop_words_param, ngram_range=(1, 2))

    # --- Modelo BERTopic configurado como en el script anterior
    topic_model = BERTopic(
        language=language if language in ["en", "es"] else "english",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True
    )

    documents = df[text_column].tolist()
    topics, _ = topic_model.fit_transform(documents, embeddings=embeddings)

    df["embedding_index"] = df.index
    df["topic"] = topics

    # # --- Visualización opcional
    # try:
    #     fig = topic_model.visualize_topics()
    #     html_path = os.path.join(output_dir, "bertopic_topics.html")
    #     fig.write_html(html_path)
    #     print(f"Gráfico interactivo guardado en: {html_path}")
    # except Exception as e:
    #     print(f"No se pudo generar la visualización: {e}")

    # --- Outliers
    outliers = df[df["topic"] == -1]
    print(f"\nOutliers detectados: {len(outliers)} documentos asignados al tópico -1.")

    return topic_model, df