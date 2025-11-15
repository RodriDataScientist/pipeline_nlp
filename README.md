# 🧠 Sistema de Análisis de Texto con Modelado de Tópicos y Reporte Interactivo

Este proyecto implementa un pipeline completo de **procesamiento, análisis y visualización de texto**, a partir de un archivo CSV.  
Incluye preprocesamiento avanzado, extracción de n-gramas, modelado de tópicos (BERTopic + UMAP + HDBSCAN), generación de embeddings y creación automática de un **reporte HTML interactivo**.

El objetivo del proyecto es permitir análisis exploratorios de texto de forma automatizada, flexible y visualmente atractiva.

---

## 🚀 Características principales

### 📥 Entrada del sistema
- Lee un archivo **CSV** que contiene una columna con texto.
- Permite especificar el nombre de la columna mediante parámetros.

---

## 🔧 Procesamiento del texto

El pipeline incluye:

### ✔️ Limpieza y normalización  
- Eliminación de acentos  
- Eliminación de puntuación y caracteres especiales  
- Normalización de espacios  
- Reducción a minúsculas  
- Eliminación de URLs, números y tokens inválidos  

### ✔️ Lematización (según idioma)
- **Inglés:** WordNetLemmatizer  
- **Español o multilingüe:** modelos spaCy (`es_core_news_sm`, `xx_ent_wiki_sm`)  
- Stopwords personalizadas según el idioma  

El resultado final se almacena en la columna:

```

processed_text

```

---

## 🧩 Embeddings

Se generan usando **SentenceTransformers**, configurable según idioma:

- `all-mpnet-base-v2`
- `paraphrase-multilingual-mpnet-base-v2`
- `distiluse-base-multilingual-cased-v1`

Los embeddings se guardan en formato `embeddings.npy`.

---

## 🔍 Modelado de tópicos (BERTopic)

El sistema implementa:

- Reducción dimensional con **UMAP (10D)**
- Clustering con **HDBSCAN**
- Modelado de tópicos optimizado para textos en español o inglés
- Identificación de documentos outliers (tópico -1)
- Generación de:
  - palabras clave por tópico
  - documento representativo por tópico
  - clusters limpios mediante ablation (eliminación de palabras redundantes)

---

## 📊 Visualizaciones generadas en el reporte

El reporte HTML incluye:

### 🔤 **Nube de palabras**
- Personalizable con distintas paletas de color
- 200 palabras máximas por defecto

### 🧩 **Top N-gramas**
- Top-10 bigramas
- Top-10 trigramas
- Gráficas con orientaciones entre 45°–90° para legibilidad
- Usan **Plotly** con patrones, colores y hover interactivo

### 🌐 **Mapa interactivo UMAP**
- Proyección 2D de embeddings
- Visualización con Bokeh
- Colores, marcadores y patrones según tópico
- Hover con texto completo

### 🏷️ **Resumen de tópicos**
- Palabras más frecuentes por tópico
- Documento representativo
- Ablación de redundancias

### ⚠️ **Outliers**
- Visualización de textos asignados al tópico -1  
- Muestra hasta 20 documentos

### 📚 **Textos más largos**
- Top-10 textos más extensos

---

## 🎨 Personalización del reporte

El usuario puede configurar:

- Título del reporte  
- Paleta de colores (4 estilos: `zesty`, `corporate`, `elegant`, `retro`)
- Modo de accesibilidad a color (`normal`, `protanopia`, `deuteranopia`)
- Número de n-gramas
- Columna de texto de entrada

---

## 🛠️ Estructura del proyecto

```
data/
text_reporter/
│
├── helpers.py                # Wordcloud, n-gramas, UMAP, representantes
├── utils.py                  # HTML template, paletas y modelos
│
├── text_preprocessing.py     # Limpieza y lematización
├── embeddings_generator.py   # Generación de embeddings
├── topic_modeling.py         # Entrenamiento BERTopic
├── report_generator.py       # Construcción del reporte HTML
│
└── main.py                   # Ejecución orquestada del pipeline

````

---

## ▶️ Ejemplo de uso

```python
import pandas as pd
from text_reporter.pipeline import preprocess_dataframe, train_topic_model, build_report
from text_reporter.embeddings import generate_embeddings

df = pd.read_csv("mis_datos.csv")

# 1. Preprocesamiento
df, model_name = preprocess_dataframe(df, text_column="comentarios", lang="multi")

# 2. Embeddings
embeddings = generate_embeddings(df, text_column="processed_text", model_name=model_name)

# 3. Modelado de tópicos
topic_model, df = train_topic_model(df, embeddings_path="embeddings.npy", language="multi")

# 4. Reporte final
build_report(df, embeddings, title="Reporte de Análisis", output_path="reporte_final.html")
````

---

## 📦 Requisitos

```
numpy
pandas
sentence-transformers
spacy
nltk
bertopic
hdbscan
umap-learn
plotly
bokeh
jinja2
wordcloud
```

---

## 🙌 Autor

Rodrigo Cervantes – Proyecto académico / profesional de análisis textual con Python.
