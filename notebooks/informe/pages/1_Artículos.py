import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Análisis de Artículos", layout="wide")

## ==========================================
## CONFIGURACIÓN DE RUTAS BASE
## ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "streamlit"

## ==========================================
## CARGA DE DATOS CACHEADA
## ==========================================
@st.cache_data
def load_data():
    cat_counts = pd.read_csv(DATA_DIR / "category_counts.csv")
    wc_idx = pd.read_csv(DATA_DIR / "wordclouds_by_category_index.csv")
    wc_temp_idx = pd.read_csv(DATA_DIR / "wordclouds_by_category_period_index.csv")
    vocab = pd.read_csv(DATA_DIR / "bow_vocabulary.csv")
    return cat_counts, wc_idx, wc_temp_idx, vocab

cat_counts, wc_idx, wc_temp_idx, vocab = load_data()

# Encabezado e introducción general
st.title("1. Análisis de los Artículos")
st.markdown("""
En este primer apartado exploramos el vocabulario de la base de datos historica de anfibia (2012 - 2026).
Para que los términos puedan compararse de forma pareja, el texto pasó por una limpieza previa: se retiraron conectores, preposiciones y modismos comunes, reduciendo las palabras a su raíz común (por ejemplo, *elecciones* o *electoral* se reúnen bajo un mismo término base).
""")

st.divider()

## ==========================================
## RESUMEN: PREPARACIÓN Y LIMPIEZA DEL CORPUS
## ==========================================
with st.expander("Ver detalles del proceso de limpieza y preparación", expanded=False):
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        st.markdown("**1. Criterios de Exclusión**")
        st.write("• **Notas vacías:** Se descartaron 26 registros sin cuerpo.")
        st.write("• **Republicaciones:** Se filtraron artículos duplicados.")
        st.write("• **Idioma:** Se conservaron únicamente publicaciones en español.")
        st.write("Cantidad de articulos:")
        # Tarjetas rápidas de volumen
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Antes", "3.622")
        with m2:
            st.metric("Despues", "3.556")

    with col_c2:
        st.markdown("**2. Unificación del Texto**")
        st.write("Para que los módelos tengan el mayor contexto posible, se los entreno con una consolidación de diferentes campos:")
        st.caption("Texto: Título + Copete + Tagline + Cuerpo")

    with col_c3:
        st.markdown("**3. Lematización (Reducción léxica)**")
        st.write("Se llevaron las palabras a su raíz común (ej. *corriendo*, *corrió* → *correr*).")
        st.write("• Palabras únicas iniciales: **~67.000**")
        st.write("• Palabras únicas lematizadas: **~43.000**")
        st.caption("Se redujeron ~24.000 variantes, lo cual facilito el entrenamiento de los módelos")

st.divider()


## ==========================================
## SECCIÓN 1: FRECUENCIA DE PALABRAS (BAG OF WORDS)
## ==========================================
import plotly.express as px

st.subheader("1. Frecuencia de Palabras (Bag of Words)")
st.write("Explorador del vocabulario normalizado del archivo. Permite analizar la cantidad total de apariciones de cada palabra y su presencia a lo largo de los distintos artículos.")

col_izq, col_der = st.columns(2)

# Mapeo y selección de columnas para visualización clara
cols_renombradas = {
    "term": "Palabra",
    "document_frequency": "Presencia en Artículos",
    "term_frequency": "Apariciones Totales"
}

df_vocab_base = vocab.rename(columns=cols_renombradas)
columnas_mostrar = [c for c in ["Palabra", "Presencia en Artículos", "Apariciones Totales"] if c in df_vocab_base.columns]

col_term = "Palabra" if "Palabra" in df_vocab_base.columns else df_vocab_base.columns[0]
col_freq = "Apariciones Totales" if "Apariciones Totales" in df_vocab_base.columns else df_vocab_base.columns[1]

with col_izq:
    df_top_plot = df_vocab_base.nlargest(20, col_freq).sort_values(by=col_freq, ascending=True)
    
    fig_bow = px.bar(
        df_top_plot,
        x=col_freq,
        y=col_term,
        orientation="h",
        text=col_freq,
        color=col_freq,
        color_continuous_scale="Blues",
        title="Top 20 Palabras Más Frecuentes"
    )
    
    fig_bow.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        cliponaxis=False
    )
    
    fig_bow.update_layout(
            title={
                "text": "Top 20 Palabras Más Frecuentes",
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top"
            },
            xaxis_title="Apariciones Totales",
            yaxis_title="",
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
            margin=dict(l=10, r=40, t=50, b=10),
            height=500
        )
    st.plotly_chart(fig_bow, use_container_width=True)

with col_der:
    palabra_busqueda = st.text_input("Buscar palabra en el vocabulario:", placeholder="Ej: justicia, política, memoria...")
    
    if palabra_busqueda.strip():
        df_vocab_vista = df_vocab_base[
            df_vocab_base[col_term].astype(str).str.contains(palabra_busqueda.strip(), case=False, na=False)
        ]
    else:
        df_vocab_vista = df_vocab_base.head(20)
        
    st.dataframe(df_vocab_vista[columnas_mostrar], use_container_width=True, height=500, hide_index=True)

## ==========================================
## SECCIÓN 2: DISTRIBUCIÓN GENERAL DE CATEGORÍAS
## ==========================================
st.subheader("2. Distribución de Artículos por Categoría")
st.write("Volumen total de artículos en la base de datos según su categoria.")
st.caption("Nota: Algunos articulos contaban con más de una categoría de forma simultánea. Para este analisis contamos individualmente cada articulo para cada categoria; por eso la suma total del gráfico es mayor a la cantidad real de artículos del archivo")

df_cat_sorted = cat_counts.rename(columns={
    "category": "Categoría",
    "documents": "Artículos"
}).sort_values(by="Artículos", ascending=True)

# Cálculo de altura dinámica según la cantidad de filas para que no se aprieten
altura_grafico = max(380, len(df_cat_sorted) * 35)

fig_cat = px.bar(
    df_cat_sorted,
    x="Artículos",
    y="Categoría",
    orientation="h",
    text="Artículos",
    color="Artículos",
    color_continuous_scale=["#93c5fd", "#2563eb", "#1e3a8a"]
)

fig_cat.update_traces(
    texttemplate="<b>%{text:,}</b>",
    textposition="outside",
    cliponaxis=False,
    marker_line_width=0,
    opacity=0.92,
    hovertemplate="<b>%{y}</b><br>Total artículos: %{x:,}<extra></extra>"
)

fig_cat.update_layout(
    title={
        "text": "Total de Artículos Publicados por Categoría",
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
        "font": {"size": 16}
    },
    xaxis_title="Cantidad de Artículos",
    yaxis_title="",
    coloraxis_showscale=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(156, 163, 175, 0.2)",
        zeroline=False
    ),
    yaxis=dict(
        tickfont={"size": 13}
    ),
    margin=dict(l=10, r=60, t=50, b=20),
    height=altura_grafico
)

st.plotly_chart(fig_cat, use_container_width=True)

st.divider()

## ==========================================
## SECCIÓN 3: PRODUCCIÓN EDITORIAL EN EL TIEMPO
## ==========================================
st.subheader("3. Artículos en en el Tiempo")
st.write("Evolución del volumen de artículos publicados por año a lo largo de los años.")

# Cargamos los atributos guardados en el pipeline
@st.cache_data
def load_attributes():
    df_attrs = pd.read_csv(DATA_DIR / "bow_document_attributes.csv", usecols=["doc_id", "post_date"])
    df_attrs["post_date"] = pd.to_datetime(df_attrs["post_date"], errors="coerce")
    df_attrs["post_year"] = df_attrs["post_date"].dt.year
    return df_attrs.dropna(subset=["post_year"])

df_attr = load_attributes()

# Agrupamos conteo por año
df_anual = (
    df_attr.groupby("post_year")
    .size()
    .reset_index(name="Artículos")
    .rename(columns={"post_year": "Año"})
)
df_anual["Año"] = df_anual["Año"].astype(int).astype(str)

fig_tiempo = px.bar(
    df_anual,
    x="Año",
    y="Artículos",
    text="Artículos",
    color="Artículos",
    color_continuous_scale=["#93c5fd", "#2563eb", "#1e3a8a"]
)

fig_tiempo.update_traces(
    texttemplate="<b>%{text}</b>",
    textposition="outside",
    cliponaxis=False,
    opacity=0.92,
    hovertemplate="<b>Año %{x}</b><br>Artículos publicados: %{y:,}<extra></extra>"
)

fig_tiempo.update_layout(
    title={
        "text": "Artículos Publicados por Año",
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
        "font": {"size": 16}
    },
    xaxis_title="Año de Publicación",
    yaxis_title="Cantidad de Artículos",
    coloraxis_showscale=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(showgrid=True, gridcolor="rgba(156, 163, 175, 0.2)", zeroline=False),
    xaxis=dict(type="category"),
    margin=dict(l=10, r=20, t=50, b=20),
    height=420
)

st.plotly_chart(fig_tiempo, use_container_width=True)

st.divider()


## ==========================================
## SECCIÓN 2: NUBES DE PALABRAS (ESTÁTICA Y TEMPORAL)
## ==========================================
st.subheader("Nubes de Palabras")
st.write("""
Representación visual del vocabulario. El tamaño de cada término es proporcional a su frecuencia de aparición.

Con ellas podremos visualizar qué palabras son las más comunes en cada categoría.
""")

st.caption("""
Nota: Únicamente creamos nubes para aquellas ventanas de tiempo que alcancen al menos 20 artículos publicados, 5.000 tokens y 80 términos únicos. Por ello, algunas combinaciónes de Categoría-Año no existen.
""")

col_ctrl1, col_ctrl2 = st.columns(2)

# Lista base ordenada por volumen histórico de mayor a menor
categorias_por_volumen = list(cat_counts["category"].unique())

## --- Controles Izquierda (Histórica) ---
with col_ctrl1:
    st.markdown("#### Nubes Históricas")
    st.caption("Se tienen en cuenta todos los artículos de una categoría.")
    categorias_hist = [c for c in categorias_por_volumen if c in wc_idx["category"].values]
    cat_elegida = st.selectbox("Elegí una categoría:", categorias_hist, key="cat_hist")

## --- Controles Derecha (Temporal) ---
with col_ctrl2:
    st.markdown("#### Nubes por Año")
    st.caption("Se tienen en cuenta solo los artículos de cierto año.")
    
    sub_col_cat, sub_col_per = st.columns(2)
    with sub_col_cat:
        # Filtramos respetando el mismo orden de volumen
        cats_con_stat = set(wc_temp_idx[wc_temp_idx["enough_statistics"] == True]["category"].unique())
        cats_temp = [c for c in categorias_por_volumen if c in cats_con_stat]
        cat_temp_elegida = st.selectbox("Categoría:", cats_temp, key="cat_temp")
    
    with sub_col_per:
        periodos_disp = sorted(wc_temp_idx[
            (wc_temp_idx["category"] == cat_temp_elegida) & 
            (wc_temp_idx["enough_statistics"] == True)
        ]["period"].unique())
        periodo_elegido = st.selectbox("Período:", periodos_disp, key="per_temp")

# --- Renderizado de Imágenes Alineadas ---
col_img1, col_img2 = st.columns(2)

with col_img1:
    if cat_elegida:
        ruta_img = BASE_DIR / wc_idx[wc_idx["category"] == cat_elegida]["path"].values[0]
        if ruta_img.exists():
            st.image(Image.open(ruta_img), use_container_width=True)
        else:
            st.error("Imagen no encontrada.")

with col_img2:
    if cat_temp_elegida and periodo_elegido:
        ruta_img_t = BASE_DIR / wc_temp_idx[
            (wc_temp_idx["category"] == cat_temp_elegida) & 
            (wc_temp_idx["period"] == periodo_elegido)
        ]["path"].values[0]
        if ruta_img_t.exists():
            st.image(Image.open(ruta_img_t), use_container_width=True)
        else:
            st.error("Imagen no encontrada.")

st.divider()

# ## ==========================================
# ## SECCIÓN 3: EXPLORADOR DE VOCABULARIO NORMALIZADO (BOW)
# ## ==========================================
# st.subheader("Vocabulario Consolidado del Archivo")
# st.write("""
# Listado de términos normalizados presentes en el archivo. Permite consultar cuántas veces se escribió una palabra en total y en cuántos artículos distintos aparece.
# """)
# st.dataframe(vocab, use_container_width=True)