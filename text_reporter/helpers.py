from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category20

import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go

import umap
import base64
import io
import re
import string
import random

# ------------- report_generator --------------
def make_wordcloud(texts, palette_colors, max_words=200):
    text = " ".join(texts)
    wc = WordCloud(
        width=1200, height=600, max_words=max_words,
        color_func=lambda *args, **kw: random.choice(palette_colors)
    ).generate(text)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def top_ngrams(texts, n=10, ngram_range=(2,2)):
    cv = CountVectorizer(ngram_range=ngram_range, max_features=20000, token_pattern=r"(?u)\b\w\w+\b")
    X = cv.fit_transform(texts)
    sums = X.sum(axis=0).A1
    items = list(zip(cv.get_feature_names_out(), sums))
    items = sorted(items, key=lambda x: x[1], reverse=True)[:n]
    return items

def plot_bar(items, title="Top", orient_angle=45, color="#000"):
    labels = [it[0] for it in items]
    values = [int(it[1]) for it in items]

    patterns = ['/', '\\', 'x', '-', '|', '+', '.']
    patterns = (patterns * ((len(labels) // len(patterns)) + 1))[:len(labels)]

    fig = go.Figure([
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=color,              
                line=dict(color="black", width=1.2),
                pattern=dict(
                    shape=patterns,     
                    solidity=0.25         
                )
            ),
            hovertemplate="<b>%{x}</b><br>Frecuencia: %{y}<extra></extra>",
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_tickangle=orient_angle if len(labels) <= 15 else 90,
        margin=dict(b=180),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#dcdcdc"),
        bargap=0.25,
    )

    return pyo.plot(fig, include_plotlyjs=False, output_type="div")

def umap_scatter(embeddings, topics, texts, palette_colors):
    if embeddings is None or len(embeddings) == 0:
        return "<p>No embeddings.</p>"

    emb = np.asarray(embeddings)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )
    emb_2d = reducer.fit_transform(emb)

    xs = emb_2d[:, 0]
    ys = emb_2d[:, 1]

    marker_types = [
        "circle", "triangle", "square", "diamond",
        "cross", "x", "star"
    ]

    hatch_patterns = [
        "/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"
    ]

    # Garantizar longitud suficiente
    def cycle(lst, idx): return lst[idx % len(lst)]

    num_colors = len(palette_colors)

    shapes = [cycle(marker_types, t) for t in topics]
    hatches = [cycle(hatch_patterns, t) for t in topics]
    point_colors = [palette_colors[t % num_colors] for t in topics]

    import textwrap
    wrapped_texts = [
        textwrap.fill(t, width=150).replace("\n", "<br>")
        for t in texts
    ]

    source = ColumnDataSource({
        "x": xs,
        "y": ys,
        "topic": topics,
        "text": wrapped_texts,
        "color": point_colors,
        "shape": shapes,
        "hatch": hatches
    })

    p = figure(
        width=900,
        height=700,
        sizing_mode="stretch_both",
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        background_fill_color="#fafafa",
        outline_line_color="#DDD",
    )

    p.hover.tooltips = """
    <div style="max-width: 260px; white-space: normal;">
        <b>Tópico:</b> @topic<br>
        <b>Texto:</b><br>
        <div style="font-size: 12px;">@text</div>
    </div>
    """

    for shape_name in set(shapes):
        subset = source.data["shape"] == shape_name

        mask = [s == shape_name for s in shapes]

        sub_source = ColumnDataSource({
            k: [v[i] for i in range(len(v)) if mask[i]]
            for k, v in source.data.items()
        })

        p.scatter(
            "x",
            "y",
            source=sub_source,
            marker=shape_name,
            size=9,
            fill_color="color",
            fill_alpha=0.8,
            line_color="#444",
            line_alpha=0.6,
            hatch_pattern="hatch",
            hatch_alpha=0.25,
            hatch_color="black"
        )

    script, div = components(p)
    return script + div

def compute_topic_representatives(df, embeddings, topic_col="topic", text_col="processed_text"):
    """
    Devuelve el documento más cercano al centroide por cada tópico (excepto -1).
    """
    representatives = {}
    valid_df = df[df[topic_col] != -1]

    for topic in sorted(valid_df[topic_col].unique()):
        cluster = valid_df[valid_df[topic_col] == topic]

        if len(cluster) == 0:
            continue

        # Seleccionar embeddings en el orden del DF
        idxs = cluster.index.tolist()
        X = embeddings[idxs]

        centroid = X.mean(axis=0, keepdims=True)
        distances = cosine_distances(X, centroid).flatten()
        best_idx = distances.argmin()

        representatives[topic] = cluster.iloc[best_idx][text_col]

    return representatives

# ------------- text_preprocessing --------------
PUNCTUATION = string.punctuation.replace("-", "") 

import unicodedata

def clean_text(text: str) -> str:
    """
    Limpieza fuerte del texto antes de la lematización.
    - Quita acentos
    - Quita puntuación
    - Quita números
    - Quita caracteres repetidos raros
    - Normaliza espacios
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", PUNCTUATION))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def lemmatize_text(text: str, lang: str, STOPWORDS, lemmatizer=None, nlp=None) -> str:
    """Lematiza texto según el idioma seleccionado (asegurando lowercasing)."""
    if not isinstance(text, str) or not text:
        return ""

    text = text.lower().strip()

    if lang == "en":
        tokens = [
            lemmatizer.lemmatize(word)
            for word in text.split()
            if word not in STOPWORDS and len(word) > 2
        ]
    else:
        doc = nlp(text)
        tokens = []
        for token in doc:
            lemma = token.lemma_.lower().strip()
            # filtra puntuación, espacios y stopwords
            if not lemma or token.is_punct or token.is_space:
                continue
            if lemma in STOPWORDS:
                continue
            if len(lemma) <= 2:
                continue
            tokens.append(lemma)

    return " ".join(tokens)