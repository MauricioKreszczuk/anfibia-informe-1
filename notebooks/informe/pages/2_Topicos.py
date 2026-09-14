import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Modelado de Tópicos - Revista Anfibia", layout="wide")

# ==========================================
# RUTAS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "streamlit"
GRAFICOS_DIR = DATA_DIR / "graficos"

# ==========================================
# CARGA DE DATOS CACHEADA
# ==========================================
@st.cache_data
def load_data():
    topicos_info = pd.read_csv(DATA_DIR / "topicos_info_centralizada.csv")
    return topicos_info

topicos_info = load_data()

def cargar_html(ruta, height=750):
    if ruta.exists():
        with open(ruta, "r", encoding="utf-8") as f:
            components.html(f.read(), height=height, scrolling=True)
    else:
        st.warning(f"Falta el gráfico interactivo: `{ruta.name}`. Verificá su exportación desde el notebook 06.5.")

# ==========================================
# ENCABEZADO Y EXPLICACIÓN METODOLÓGICA
# ==========================================
st.title("2. Modelado de Tópicos")

st.markdown("""
Consiste en el agrupamiento automático de los artículos según la similitud de su contenido.  

El algoritmo que utilizamos (BERTopic) lee los textos de los articulos y los reúne según su parecido temático. Luego, a cada grupo le asigna las palabras clave que mejor resumen de qué se habló en esas publicaciones.
""")
with st.expander("Detalles del modelo", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Clustering y Embeddings Naturales**")
        st.write("""
        • **Generación de Embeddings:** Se procesan los cuerpos de los articulos con `paraphrase-multilingual-MiniLM-L12-v2`, generando la representación vectorial de cada articulo, que se utilizara para decidir el agrupamiento (Clustering).
        
        • **Tratamiento de outliers (-1):** Más de 1600 articulos quedaban bajo el Tópico -1 (reservado para aquellas que el modelo no supo agrupar con las demas). Se reasignaron a su tópico más cercano calculando similitud coseno contra los centroides vectoriales de cada Tópico.
        """)
    with c2:
        st.markdown("**2. Descriptores**")
        st.write("""
        • **Palabras Clave:** Se extraen sobre el texto unificado, saltando las palabras redundantes.
        """)

    st.caption("""
    Recordatorio: 
    
    Texto unificado: Título + Copete + Tagline + Cuerpo
    """)

st.divider()

# ==========================================
# SECCIÓN 1: TABLA CENTRALIZADA DE TÓPICOS
# ==========================================
st.subheader("1. Tópicos Encontrados")
st.write("Estos son los tópicos resultantes del modelo y sus características principales.")

# Ajuste visual de columnas para legibilidad
column_config = {
    "Topic": st.column_config.NumberColumn("ID Tópico", format="%d"),
    "Name": st.column_config.TextColumn("Palabras Clave"),
    "Docs": st.column_config.NumberColumn("Artículos", format="%d"),
    "% Corpus": st.column_config.NumberColumn("% Artículos", format="%.2f %%"),
    "Articulos_Representativos": st.column_config.TextColumn("Artículo Representativo"),
    "Tagline_Representativo": st.column_config.TextColumn("Tagline"),
    "URL_Articulo": st.column_config.LinkColumn(
        "Enlace",
        display_text="Ver nota ↗"
    )
}

columnas_visibles = [
    "Topic", 
    "Name", 
    "Docs", 
    "% Corpus", 
    "Articulos_Representativos", 
    "Tagline_Representativo",
    "URL_Articulo"
]

df_tabla = topicos_info[[c for c in columnas_visibles if c in topicos_info.columns]]

st.dataframe(
    df_tabla,
    use_container_width=True,
    hide_index=True,
    column_config=column_config
)

st.caption("""
Nota: Se determinaron los artículos clave de cada tópico (junto con su bajada o tagline) calculando el centroide vectorial de cada tema (el "promedio semántico" de todos sus textos) y seleccionando las publicaciones con mayor similitud coseno respecto a ese centro. Representan el núcleo del tema, no necesariamente los artículos más leídos.
""")

st.divider()

# ==========================================
# SECCIÓN 2: PALABRAS CLAVE DE LOS TÓPICOS
# ==========================================
st.subheader("2. Palabras Clave de los Tópicos Principales (Top 16)")
st.write("Observamos el peso de cada palabra clave sobre el tópico y cómo estas definen su identidad.")

cargar_html(GRAFICOS_DIR / "01_barchart_topicos.html", height=850)

# ----------------------------------------------------
# Ficha de consulta para tópicos fuera del Top 16
# ----------------------------------------------------
st.markdown("#### Detalles:")
#st.write("Para explorar temas específicos que no ingresaron en los 16 principales de arriba:")

opciones_topicos = {
    row["Topic"]: f"Tópico {row['Topic']} — {row['Name']}"
    for _, row in topicos_info.iterrows()
    if row["Topic"] != -1
}

topico_sel = st.selectbox(
    "Seleccioná un tópico:",
    options=list(opciones_topicos.keys()),
    format_func=lambda x: opciones_topicos[x],
    label_visibility="collapsed"
)

fila = topicos_info[topicos_info["Topic"] == topico_sel].iloc[0]
palabras = [p.strip() for p in str(fila.get("Name", "")).split(",") if p.strip()]

c1, c2, c3 = st.columns([1, 1, 3])
c1.metric("Volumen", f"{int(fila.get('Docs', 0))} notas")
c2.metric("Volumen (%)", f"{float(fila.get('% Corpus', 0)):.2f} %")
with c3:
    st.caption("Palabras clave del tópico:")
    st.pills("Términos", palabras, selection_mode="multi", disabled=True, label_visibility="collapsed")

# Artículo representativo del tópico seleccionado
art = fila.get("Articulos_Representativos", "")
tag = fila.get("Tagline_Representativo", "")
url = fila.get("URL_Articulo", "")

if pd.notna(art) and str(art).strip():
    st.markdown(f"**Artículo representativo:** {art}")
    if pd.notna(tag) and str(tag).strip() and str(tag).lower() != "nan":
        st.caption(f"_{tag}_")
    if pd.notna(url) and str(url).strip() and str(url).startswith("http"):
        st.link_button("Ver Nota ↗", str(url).strip())

st.divider()

# ==========================================
# SECCIÓN 3: EVOLUCIÓN TEMPORAL Y COYUNTURA
# ==========================================
st.subheader("3. Evolución Temporal de los Tópicos")
st.write("""
Graficamos la frecuencia de cada Tópico según su fecha.
Los marcadores en forma de diamante (◆) señalan el **máximo histórico** de cada tópico.
""")
st.caption("Al pasar el cursor sobre los picos, se muestran los articulos representativos de ese pico." \
"Se pueden seleccionar los Tópicos a mostrar en las leyendas de la derecha.")

cargar_html(GRAFICOS_DIR / "02_evolucion_topicos_completa.html", height=780)


with st.expander("Detalles: Metodológia para calcular los articulos de cada pico"):
    st.markdown("""
    Se aplicó el siguiente criterio:

    * **Ventanas temporales dinámicas (*bins*):** Se segmentaron los articulos en intervalos de tiempo mediante *Dynamic Topic Modeling*, agrupando el volumen de articulos en cada ventana.
    * **Identificación de picos y máximos:** 
        * **Máximo Histórico (◆):** El punto temporal con el volumen absoluto más alto de publicaciones del tópico.
        * **Picos locales:** Momentos donde la cantidad de publicaciones superó al período previo y al posterior.
    * **Artículos representativos:** Para cada período se calcularon los embeddings de los articulos publicados en esa ventana y se obtuvo su **centroide semántico local** (el promedio vectorial del intervalo).
    Mediante similitud coseno contra dicho centroide, se extrajo el *Top 3* de publicaciones más cercanas, buscando explicar que notica ocasiono el pico.
    """)

st.divider()

# ==========================================
# SECCIÓN 4: ESPACIO SEMÁNTICO (UMAP)
# ==========================================
st.subheader("4. Mapa de los artículos (Proyección 2D)")
st.write("""
Proyección bidimensional de los artículos.
Cada punto representa un artículo publicado, su ubicacion esta dada por el vector representativo de cada artículo (embeddings). Entonces los articulos que sean cercanos seran similares (porque sus representaciónes vectoriales lo son).
Los colores corresponden a los tópicos del modelo.
""")
st.caption("Seleccioná un área para hacer zoom y hacé doble clic sobre el gráfico para reestablecer la vista. Usá la leyenda de la derecha para encender, apagar o aislar Tópicos.")
st.caption("Posa el mouse sobre un articulo para ver su Tópico, nombre y tagline correspondiente.")

cargar_html(GRAFICOS_DIR / "03_espacio_documentos.html", height=750)

with st.expander("Detalles: Como se construyo el mapa"):
    st.markdown("""
    * **Espacio vectorial original:** Cada articulo se convierte en un vector (embedding), que representara su contexto. Pero, con muchas dimensiones (cientos de variables).
    * **Reducción dimensional (UMAP):** Para poder graficarlo en dos dimensiones, se utiliza el algoritmo **UMAP** (*Uniform Manifold Approximation and Projection*). Reduciendo la dimensinalidad de los vectores a 2, sin perder sus relaciónes.
    """)