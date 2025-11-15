import numpy as np
import torch

from sentence_transformers import SentenceTransformer

def generate_embeddings(df, text_column="processed_text", model_name=None, output_path="embeddings.npy"):
    """
    Genera embeddings según el modelo especificado.

    Args:
        df (pd.DataFrame): DataFrame con los textos limpios.
        text_column (str): Columna con texto preprocesado.
        model_name (str): Nombre del modelo de SentenceTransformers.
        output_path (str): Ruta del archivo .npy para guardar embeddings.
    """
    model_name = model_name or "paraphrase-multilingual-mpnet-base-v2"
    print(f"\n🧩 Cargando modelo de embeddings: {model_name}")
    model = SentenceTransformer(model_name)

    texts = df[text_column].tolist()
    print(f"Generando embeddings para {len(texts)} documentos...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=16,
        device=device
    )

    np.save(output_path, embeddings)
    print(f"\n✅ Embeddings guardados en: {output_path}")
    print(f"Dimensiones: {embeddings.shape[1]} | Forma: {embeddings.shape}")

    return embeddings