# --------------------- CONFIGURACIÓN DE MODELOS -----------------------
SPACY_MODELS = {
    "es": "es_core_news_sm",      # Español
    "en": "en_core_web_sm",       # Inglés
    "multi": "xx_ent_wiki_sm"     # Multilingüe
}

EMBEDDING_MODELS = {
    "es": "distiluse-base-multilingual-cased-v1",
    "en": "all-mpnet-base-v2",
    "multi": "paraphrase-multilingual-mpnet-base-v2"
}

# --------------------- COLORES -----------------------
COLOR_PALETTES = {
    "zesty": {
        "normal": ["#F5793A", "#A95AA1", "#85CDF9", "#0F2080"],
        "protanopia": ["#AEC545", "#6073B1", "#A8BFB8", "#052955"],
        "deuteranopia": ["#C59434", "#6F7498", "#A3B7F9", "#092C48"],
    },
    "corporate": {
        "normal": ["#B0B8AD", "#EBE7E0", "#C5D4E1", "#44749D"],
        "protanopia": ["#B0B6AB", "#EDE6DE", "#D1DDDE", "#636D97"],
        "deuteranopia": ["#CDB1AD", "#FADFE2", "#DCEBE3", "#5D6E9E"],
    },
    "elegant": {
        "normal": ["#ABC3C9", "#E6DCD3", "#CCB69F", "#382119"],
        "protanopia": ["#BEBCC5", "#E2DAD1", "#C9BD9E", "#2E2B21"],
        "deuteranopia": ["#CABBCB", "#F4D4D4", "#DCB69F", "#3A242F"],
    },
    "retro": {
        "normal": ["#601AAA", "#EE442F", "#63C8CE", "#F9F4EC"],
        "protanopia": ["#2A3B58", "#8B7F47", "#9C9EB5", "#FAF2EA"],
        "deuteranopia": ["#383745", "#A17724", "#9E9CC2", "#FDF0F2"],
    }
}

# --------------------- HTML TEMPLATE -----------------------
HTML_TEMPLATE = """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>{{ title }}</title>

  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.3.0.min.js"></script>
  <link rel="stylesheet" href="https://cdn.bokeh.org/bokeh/release/bokeh-3.3.0.min.css">

  <script src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.3.0.min.js"></script>
  <link rel="stylesheet" href="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.3.0.min.css">

  <script src="https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.3.0.min.js"></script>
  <link rel="stylesheet" href="https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.3.0.min.css">

  <style>

    /* ----------- RESET & BASE ----------- */
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
      background: {{ secondary }};
      color: {{ dark }};
      line-height: 1.55;
    }

    h1, h2, h3 {
      font-weight: 600;
      letter-spacing: -0.5px;
    }

    /* ----------- HEADER ENHANCED ----------- */
    header {
      background: {{ primary }};
      color: white;
      padding: 32px 50px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      border-bottom: 8px solid {{ accent }};
    }

    header h1 {
      font-size: 36px;
      margin-bottom: 6px;
    }

    header .sub {
      opacity: 0.85;
      font-size: 15px;
    }

    /* ----------- MAIN LAYOUT ----------- */
    main {
      padding: 40px 50px;
      max-width: 1500px;
      margin: auto;
    }

    /* ----------- SECTION TITLES ----------- */
    .section h3 {
      color: {{ primary }};
      margin-bottom: 10px;
      font-size: 24px;
      border-bottom: 3px solid {{ accent }};
      padding-bottom: 4px;
      font-weight: 600;
    }

    .section { 
      margin-bottom: 55px; 
    }

    /* ----------- CARD IMPROVED ----------- */
    .card {
      background: white;
      border-radius: 18px;
      padding: 28px;
      border-left: 8px solid {{ primary }};
      box-shadow: 0 4px 16px rgba(0,0,0,0.08);
      transition: transform 0.15s ease, box-shadow 0.2s ease;
    }

    .card:hover {
      transform: translateY(-3px);
      box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }

    /* ----------- FLEX GRID ----------- */
    .flex {
      display: grid;
      grid-template-columns: minmax(350px, 1fr) minmax(350px, 0.7fr);
      gap: 28px;
    }

    /* ----------- SCROLL BOX IMPROVED ----------- */
    .scroll-box {
      max-height: 350px;
      overflow-y: auto;
      background: {{ secondary }};
      padding: 16px 26px;
      border-radius: 12px;
      border: 1px solid {{ accent }};
      box-shadow: inset 0 1px 4px rgba(0,0,0,0.06);
    }

    .scroll-box::-webkit-scrollbar { width: 10px; }
    .scroll-box::-webkit-scrollbar-thumb {
      background: {{ primary }};
      border-radius: 6px;
    }

    /* ----------- PLOT WRAPPER ----------- */
    .plot-box {
      width: 100%;
      height: 680px;
      border: 2px solid {{ accent }};
      border-radius: 16px;
      background: white;
      padding: 16px;
      box-shadow: inset 0 2px 6px rgba(0,0,0,0.05);
    }

    /* ----------- REPRESENTATIVE DOC ----------- */
    .topic-rep {
      margin-top: 10px;
      padding: 12px;
      background: {{ secondary }};
      border-left: 6px solid {{ accent }};
      border-radius: 10px;
    }

    /* ----------- FOOTER ----------- */
    footer {
      text-align: center;
      padding: 24px;
      background: {{ primary }};
      color: white;
      margin-top: 50px;
      font-size: 15px;
      border-top: 8px solid {{ accent }};
    }

    ol li { margin-bottom: 8px; }

  </style>
</head>

<body>

  <!-- HEADER -->
  <header>
    <h1>{{ title }}</h1>
    <div class="sub">
      <strong>Fuente:</strong> {{ input_path }} &nbsp; | &nbsp;
      <strong>Docs:</strong> {{ n_docs }}
    </div>
  </header>

  <main>

    <!-- WORDCLOUD + TOP DOCS -->
    <div class="section flex">
      <div class="card">
        <h3>Nube de palabras (global)</h3>
        <img src="{{ wordcloud }}" style="width:100%;border-radius:12px;">
      </div>

      <div class="card">
        <h3>Top documentos (longitud)</h3>
        <div class="scroll-box">
          <ol>
          {% for d in top_docs %}
            <li><small>{{ d }}</small></li>
          {% endfor %}
          </ol>
        </div>
      </div>
    </div>

    <!-- NGRAMAS -->
    <div class="section card">
      <h3>Top n-gramas</h3>
      <div>{{ bigrams_div|safe }}</div>
      <div>{{ trigrams_div|safe }}</div>
    </div>

    <!-- TOPICS -->
    <div class="section card">
      <h3>Tópicos (BERTopic)</h3>
      <div class="scroll-box">
        {% for t, words in topics %}
          <div style="margin-bottom:18px;">
            <strong>Tópico {{ t }}:</strong> {{ words }}
            {% if t in topic_reps %}
              <div class="topic-rep">
                <em>Documento representativo:</em><br>
                <small>{{ topic_reps[t] }}</small>
              </div>
            {% endif %}
          </div>
        {% endfor %}
      </div>
    </div>

    <!-- ABLACIÓN -->
    <div class="section card">
      <h3>Ablación de tópicos (palabras únicas por tópico)</h3>
      <div class="scroll-box">
        {% for t, words in ablated %}
          <div style="margin-bottom:10px;"><strong>Tópico {{ t }}:</strong> {{ words }}</div>
        {% endfor %}
      </div>
    </div>

    <!-- UMAP -->
    <div class="section card">
      <h3>Scatter UMAP coloreado por tópico</h3>
      <div class="plot-box">
        {{ umap_div|safe }}
      </div>
    </div>

    <!-- OUTLIERS -->
    <div class="section card">
      <h3>Outliers (tópico -1)</h3>
      <div class="scroll-box">
        <ol>
        {% for o in outliers %}
          <li><small>{{ o }}</small></li>
        {% endfor %}
        </ol>
      </div>
    </div>

  </main>

  <footer>
    Reporte generado automáticamente.
  </footer>

</body>
</html>
"""