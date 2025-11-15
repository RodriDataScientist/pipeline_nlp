import os

from collections import Counter
from jinja2 import Template

from text_reporter.utils import HTML_TEMPLATE, COLOR_PALETTES
from text_reporter.helpers import (
    make_wordcloud,
    top_ngrams,
    plot_bar,
    umap_scatter,
    compute_topic_representatives
)

# ------------------ Main function ------------------
def build_report(
    df,
    embeddings,
    title="Reporte",
    text_col="processed_text",
    input_path="data.csv",
    palette=1,
    colorblind="normal",
    ngram_top=10,
    output_path="reporte_final.html"
):
    """
    Genera archivo HTML con todas las visualizaciones.
    df: DataFrame que contiene 'topic' y 'topic_probability' y la columna de texto.
    embeddings: np.ndarray con embeddings (misma orden que df)
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n_docs = len(df)
    texts = df[text_col].astype(str).tolist()

    palette_colors = COLOR_PALETTES[palette][colorblind]
    primary = palette_colors[0]
    secondary = palette_colors[1]
    accent = palette_colors[2]
    dark = palette_colors[3]

    # Wordcloud
    wc = make_wordcloud(texts, palette_colors)

    # n-gramas
    bigrams = top_ngrams(texts, n=ngram_top, ngram_range=(2,2))
    trigrams = top_ngrams(texts, n=ngram_top, ngram_range=(3,3))
    bigrams_div = plot_bar(bigrams, title="Top bigramas", orient_angle=60, color=primary)
    trigrams_div = plot_bar(trigrams, title="Top trigramas", orient_angle=60, color=primary)

    # Topics summary (top words por tópico)
    topic_reps = {}
    if "topic" in df.columns:
        topics_grouped = {}
        # Documento representativo por tópico
        topic_reps = compute_topic_representatives(df, embeddings, topic_col="topic", text_col=text_col)
        df_valid = df[df["topic"] != -1]

        for t, grp in df_valid.groupby("topic"):
            words = " ".join(grp[text_col].astype(str)).split()
            top_words = [w for w, c in Counter(words).most_common(12)]
            topics_grouped[t] = top_words
    else:
        topics_grouped = {}

    # Ablation (greedy)
    ablated = []
    assigned = set()
    for t in sorted(topics_grouped.keys()):
        words = [w for w in topics_grouped[t] if w not in assigned]
        if len(words) == 0 or all(len(w.strip()) <= 1 for w in words):
            continue  # saltar tópicos sin palabras válidas
        ablated.append((t, words[:12]))
        assigned.update(words)

    # UMAP scatter (2D)
    topics = df["topic"].tolist() if "topic" in df.columns else [0]*n_docs
    umap_div = umap_scatter(embeddings, topics, texts, palette_colors)

    # Outliers
    outlier_texts_raw = df[df["topic"] == -1][text_col].astype(str).head(20).tolist()

    # Eliminar duplicados manteniendo el orden
    outlier_texts = list(dict.fromkeys(outlier_texts_raw))

    # Top docs by length
    top_docs = df[text_col].astype(str).sort_values(key=lambda s: s.str.len(), ascending=False).head(10).tolist()

    # Prepare context and write HTML
    context = {
        "title": title,
        "input_path": input_path,
        "text_col": text_col,
        "n_docs": n_docs,
        "wordcloud": wc,
        "bigrams_div": bigrams_div,
        "trigrams_div": trigrams_div,
        "topics": [(t, ", ".join(topics_grouped[t][:10])) for t in sorted(topics_grouped.keys())],
        "ablated": [(t, ", ".join(words)) for t, words in ablated],
        "umap_div": umap_div,
        "outliers": outlier_texts,
        "top_docs": top_docs,
        "topic_reps": topic_reps,
        "primary": primary,
        "secondary": secondary,
        "accent": accent,
        "dark": dark,
    }

    html = Template(HTML_TEMPLATE).render(**context)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Reporte generado: {output_path}")
    return output_path