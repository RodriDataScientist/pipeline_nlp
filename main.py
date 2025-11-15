import os
import argparse
import pandas as pd
import numpy as np

from text_reporter.text_preprocessing import preprocess_dataframe
from text_reporter.embeddings_generator import generate_embeddings
from text_reporter.topic_modeling import train_topic_model
from text_reporter.report_generator import build_report

def run_pipeline(
    input_csv,
    text_column,
    lang="multi",
    output_dir="output",
    palette="zesty",
    colorblind="normal",    
    title="Reporte BERTopic",
    embeddings_path=None,
    n_topics=None,
    ngram_top=10
):

    os.makedirs(output_dir, exist_ok=True)
    
    # 1) Cargar CSV
    print("Cargando CSV...")
    df = pd.read_csv(input_csv)

    # 2) Preprocesamiento
    print(f"Preprocesando textos (limpieza y lematización) para idioma '{lang}'...")
    df, emb_model = preprocess_dataframe(df, text_column=text_column, lang=lang)

    if df.empty:
        raise ValueError("❌ No hay textos válidos después del preprocesamiento. Verifica que la columna contenga texto útil.")
    else:
        print(f"✅ {len(df)} textos válidos después del preprocesamiento.")

    # 3) Embeddings
    embeddings_path = embeddings_path or os.path.join(output_dir, f"embeddings_{title}.npy")
    print("Generando embeddings (sentence-transformers)...")
    embeddings = generate_embeddings(
        df,
        text_column="processed_text",
        model_name=emb_model,         
        output_path=embeddings_path
    )

    # 4) Topic modeling (BERTopic)
    print("Entrenando BERTopic...")
    topic_model, df_topics = train_topic_model(
        df,
        text_column="processed_text",
        embeddings_path=embeddings_path,
        language=lang
    )

    # 5) Reordenar embeddings si es necesario
    if "embedding_index" in df_topics.columns:
        idxs = df_topics["embedding_index"].astype(int).tolist()
        embeddings_ordered = np.load(embeddings_path)[idxs]
    else:
        embeddings_ordered = np.load(embeddings_path)

    # 6) Generar reporte HTML final
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
    out_html = os.path.join(output_dir, f"{safe_title}.html")
    print("Generando reporte HTML final...")
    build_report(
        df_topics,
        embeddings_ordered,
        title=title,
        text_col="processed_text",
        input_path=input_csv,
        palette=palette,
        colorblind=colorblind,
        ngram_top=ngram_top,
        output_path=out_html
    )

    print("\n✅ Pipeline finalizado correctamente.")
    print(f"📄 Reporte generado en: {out_html}")
    return out_html

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline completo para análisis de texto:\n"
            "  - Preprocesamiento\n"
            "  - Generación de embeddings\n"
            "  - BERTopic\n"
            "  - N-gramas, wordcloud, tópicos y visualizaciones\n"
            "  - Reporte HTML interactivo"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ------------------------
    # Grupo: Entrada
    # ------------------------
    input_group = parser.add_argument_group("Entrada de datos")
    input_group.add_argument("--input", "-i", required=True,
                             help="Ruta al archivo CSV de entrada.")
    input_group.add_argument("--text-column", "-c", required=True,
                             help="Nombre de la columna que contiene el texto a procesar.")
    input_group.add_argument("--lang", "-l", default="multi",
                             choices=["en", "es", "multi"],
                             help="Idioma para el preprocesamiento del texto.")
    input_group.add_argument("--output-dir", "-o", default="reports",
                             help="Directorio donde se guardará el reporte y los modelos.")

    # ------------------------
    # Grupo: Personalización visual
    # ------------------------
    visual_group = parser.add_argument_group("Personalización visual")
    visual_group.add_argument("--title", "-t", default="Reporte BERTopic",
                              help="Título del reporte final.")
    visual_group.add_argument("--palette", type=str, default="zesty",
                              choices=["zesty", "corporate", "elegant", "retro"],
                              help="Paleta de colores base para las visualizaciones.")
    visual_group.add_argument("--colorblind", type=str, default="normal",
                              choices=["normal", "protanopia", "deuteranopia"],
                              help="Adaptación de la paleta para accesibilidad visual.")

    # ------------------------
    # Grupo: Análisis opcionales
    # ------------------------
    analysis_group = parser.add_argument_group("Opciones de análisis")
    analysis_group.add_argument("--ngrams", type=int, default=10,
                                help="Número de bigramas y trigramas a mostrar en el reporte.")
    analysis_group.add_argument("--skip-wordcloud", action="store_true",
                                help="Desactiva la generación de la nube de palabras.")
    analysis_group.add_argument("--topic-ablation", action="store_true",
                                help="Activa la ablación de tópicos (elimina palabras redundantes entre tópicos).")
    analysis_group.add_argument("--show-outliers", action="store_true",
                                help="Incluye el análisis de outliers (tópico -1) en el reporte.")
    analysis_group.add_argument("--n-topics", type=int, default=None,
                                help="Número de tópicos a forzar para BERTopic (opcional).")

    parser.epilog = (
        "Ejemplos de uso:\n"
        "  python main.py -i data/docs.csv -c Review -l en -o reports -t Reporte Automático "
        "  --palette elegant --colorblind normal --ngrams 20 --topic-ablation --show-outliers"
    )

    args = parser.parse_args()

    run_pipeline(
        input_csv=args.input,
        text_column=args.text_column,
        lang=args.lang,
        output_dir=args.output_dir,
        title=args.title,
        palette=args.palette,
        colorblind=args.colorblind,
        n_topics=args.n_topics,
        ngram_top=args.ngrams
    )