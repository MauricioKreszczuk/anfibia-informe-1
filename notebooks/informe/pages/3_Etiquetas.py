import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import math

st.set_page_config(page_title="Etiquetas", layout="wide")

## ==========================================
## RUTAS Y CONFIGURACIÓN BASE
## ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "streamlit"
GRAFICOS_DIR = DATA_DIR / "graficos"


## ==========================================
## CARGA DE DATOS CACHEADA
## ==========================================
@st.cache_data
def load_data():
    tags_global = pd.read_csv(DATA_DIR / "tag_counts_global.csv")
    df_tags_cat = pd.read_csv(DATA_DIR / "df_tags_vs_categorias_base.csv")
    df_tags_top = pd.read_parquet(DATA_DIR / "df_tags_vs_topicos_base.parquet")
    tags_solapamiento = pd.read_csv(DATA_DIR / "tags_solapamiento_prob_topicos.csv")
    tags_auto = pd.read_csv(DATA_DIR / "tags_fusiones_automaticas.csv")
    tags_manual = pd.read_csv(DATA_DIR / "tags_revision_manual.csv")
    df_titulos = pd.read_csv(DATA_DIR / "diccionario_titulos_url.csv")
    
    # Casteo explícito de tipos antes del cruce
    df_titulos["ID"] = df_titulos["ID"].astype(int)
    df_tags_cat["ID"] = df_tags_cat["ID"].astype(int)
    
    df_tags_cat = df_tags_cat.merge(df_titulos, on="ID", how="left")
    
    # Fallback por si algún registro no tenía source_url
    if "source_url" not in df_tags_cat.columns:
        df_tags_cat["source_url"] = "https://revistaanfibia.com/?p=" + df_tags_cat["ID"].astype(str)
        
    return tags_global, df_tags_cat, df_tags_top, tags_solapamiento, tags_auto, tags_manual

(tags_global, df_tags_cat, df_tags_top, tags_solapamiento, tags_auto, tags_manual) = load_data()

def cargar_html(ruta, height=750):
    if ruta.exists():
        with open(ruta, "r", encoding="utf-8") as f:
            components.html(f.read(), height=height, scrolling=True)
    else:
        st.warning(f"Falta el gráfico: {ruta.name}")

## ==========================================
## ENCABEZADO PRINCIPAL
## ==========================================
st.title("Diagnóstico y Auditoría de Etiquetas")
st.markdown("""
Esta sección examina la cobertura de las etiquetas, su coherencia frente a las categorías y los tópicos descubiertos, junto con propuestas de consolidación.
""")

st.divider()


## ==========================================
## SECCIÓN 1: RANKING GLOBAL Y COBERTURA TEMPORAL
## ==========================================
st.subheader("1. Exploración general de las etiquetas")
st.write("""
Analizamos cómo están distribuidas las etiquetas, cuáles son las más comunes y qué artículos las utilizan.
""")

# 1. Top 100 para la tabla del ranking
df_tags_display = (
    tags_global.head(100)
    .copy()
    .reset_index(drop=True)
)
df_tags_display.columns = ["Etiqueta", "Frecuencia"]
df_tags_display.insert(0, "Ranking", range(1, len(df_tags_display) + 1))

# 2. Todas las etiquetas únicas ordenadas para el buscador
todas_las_etiquetas = sorted(tags_global[tags_global.columns[0]].dropna().unique().tolist())

# Proporción 1 a 1.4 para balance horizontal
col_ranking, col_articulos = st.columns([1, 1.4], gap="large")

with col_ranking:
    st.markdown("##### Top 100 Etiquetas")
    st.caption("Etíquetas más utilizadas.")
    st.dataframe(
        df_tags_display,
        use_container_width=True,
        hide_index=True,
        height=380,
        column_config={
            "Ranking": st.column_config.NumberColumn("Posición", width="small", alignment="center"),
            "Etiqueta": st.column_config.TextColumn("Etiqueta", width="medium", alignment="left"),
            "Frecuencia": st.column_config.ProgressColumn(
                "Cantidad de Notas",
                format="%d",
                min_value=0,
                max_value=int(df_tags_display["Frecuencia"].max()),
            ),
        }
    )

with col_articulos:
    st.markdown("##### Artículos por Etiqueta")
    
    # Buscador con autocompletado sobre todo el universo de etiquetas
    tag_seleccionado = st.selectbox(
        "Buscar etiqueta:",
        options=todas_las_etiquetas,
        index=todas_las_etiquetas.index(df_tags_display["Etiqueta"].iloc[0]) if df_tags_display["Etiqueta"].iloc[0] in todas_las_etiquetas else 0,
        placeholder="Escribí para buscar sugerencias...",
        label_visibility="collapsed"
    )

    if tag_seleccionado:
        notas_tag = (
            df_tags_cat[df_tags_cat["tag"] == tag_seleccionado]
            .drop_duplicates(subset=["ID"])
            .sort_values(by="post_year", ascending=False)
            .reset_index(drop=True)
        )
        
        df_notas_display = pd.DataFrame({
            "Año": notas_tag["post_year"],
            "Título": notas_tag["title_text"].fillna("Sin título"),
            "Categoría": notas_tag["categoria_normalizada"],
            "Enlace": notas_tag["source_url"]
        })

        st.dataframe(
            df_notas_display,
            use_container_width=True,
            hide_index=True,
            height=380,
            column_config={
                "Año": st.column_config.NumberColumn("Año", format="%d", width="small", alignment="center"),
                "Título": st.column_config.TextColumn("Título", width="large"),
                "Categoría": st.column_config.TextColumn("Categoría", width="small"),
                "Enlace": st.column_config.LinkColumn("Enlace", display_text="Ver", width="small", alignment="center")
            }
        )
    else:
        st.info("Seleccioná o buscá una etiqueta para listar sus artículos.")

    
st.caption("Si buscamos por ejemplo Religión, ya con escribir Reli nos aparecen dos etiquetas que deberian ser la misma, esto se explora más adelante.")

st.markdown("#### Cobertura de las Etiquetas")
st.write("Proporción de artículos que cuentan con al menos una etiqueta frente a los que no.")
cargar_html(GRAFICOS_DIR / "07_pie_cobertura.html", height=420)

st.markdown("Practicamente la mitad de los artículos no llevan ninguna etiqueta. Queriendo ahondar en eso, nos preguntamos si era algo común en el tiempo.")

st.markdown("#### Evolución del Etiquetado en el tiempo")
st.write("Volumen anual de los artículos con y sin etiquetas, detallando la tasa de cobertura año a año.")
cargar_html(GRAFICOS_DIR / "07_evolucion_etiquetado.html", height=520)  


st.divider()


## ==========================================
## SECCIÓN 2: ETIQUETAS Y CATEGORÍAS EDITORIALES
## ==========================================
st.subheader("2. Etiquetas vs. Categorías")
st.markdown("Cruzamos las categorías con las etiquetas para ver cómo se relaciónan:")


st.markdown("#### Distribución de Etiquetas por Categoria")
st.write("""
Volumen de etiquetas únicas asignadas dentro de cada categoría.
""")
cargar_html(GRAFICOS_DIR / "07_distribucion_etiquetas_por_categoria.html", height=450)

st.markdown("#### Etiquetas vs. Categorías (Heatmap)")
st.write("""
Muestra la cantidad de veces que una etiqueta se repite en articulos de cierta categoria. Queriamos distinguir etiquetas que se usen para más de una categoria.
""")
st.caption("Los valores indican la cantidad de artículos únicos en cada intersección.")

cats_disp = df_tags_cat["categoria_normalizada"].unique()
cats_sel = st.multiselect("Categorías a mostrar:", cats_disp, default=cats_disp)
top_tags_hm = st.slider("Cantidad de etiquetas a mostrar (ordenadas de mayor a menor cantidad de artículos):", 5, 50, 15)

if cats_sel:
    top_tags_list = df_tags_cat.groupby("tag")["ID"].nunique().nlargest(top_tags_hm).index
    df_hm = df_tags_cat[(df_tags_cat["tag"].isin(top_tags_list)) & (df_tags_cat["categoria_normalizada"].isin(cats_sel))]
    
# 1. Obtenemos las top etiquetas (de mayor a menor)
    top_tags_list = df_tags_cat.groupby("tag")["ID"].nunique().nlargest(top_tags_hm).index.tolist()
    
    df_hm = df_tags_cat[(df_tags_cat["tag"].isin(top_tags_list)) & (df_tags_cat["categoria_normalizada"].isin(cats_sel))]
    
    if not df_hm.empty:
        matriz = pd.crosstab(df_hm["tag"], df_hm["categoria_normalizada"])
        
        # Invertimos la lista para que la más frecuente quede abajo de todo en la matriz
        orden_abajo = top_tags_list[::-1]
        matriz = matriz.reindex(index=[t for t in orden_abajo if t in matriz.index])

        fig_hm = px.imshow(
            matriz,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(x="Categoría", y="Etiqueta", color="Artículos")
        )

        # Corregimos el hover y aseguramos que no invierta el eje
        fig_hm.update_traces(
            hovertemplate="<b>Etiqueta:</b> %{y}<br><b>Categoría:</b> %{x}<br><b>Artículos:</b> %{z}<extra></extra>"
        )
        fig_hm.update_yaxes(autorange=True)

        st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("#### Distribución de etiquetas en categorias en el tiempo")
st.write("""
Una vez más mostramos las etiquetas más comúnes de una categoria, pero ahora agregandole la dimensión del tiempo. Con esto seria mas claro visualizar que etiquetas marcaron cada categoria en cada año.
""")

anio_sel = st.selectbox(
    "Período a visualizar:",
    ["Global"] + sorted([str(int(a)) for a in df_tags_cat["post_year"].dropna().unique()])
)
top_dona = st.slider("Cantidad de etiquetas por gráfico:", 3, 15, 7)

if anio_sel == "Global":
    datos_dona = df_tags_cat.groupby(["categoria_normalizada", "tag"]).size()
else:
    datos_dona = df_tags_cat[df_tags_cat["post_year"] == int(anio_sel)].groupby(["categoria_normalizada", "tag"]).size()

cats_unicas = df_tags_cat["categoria_normalizada"].unique()
n_cols = 3
n_rows = math.ceil(len(cats_unicas) / n_cols)

def wrap_label(txt, max_len=14):
    if len(txt) > max_len and " " in txt:
        partes = txt.split(" ")
        medio = len(partes) // 2
        return " ".join(partes[:medio]) + "<br>" + " ".join(partes[medio:])
    return txt

fig_donas = make_subplots(
    rows=n_rows,
    cols=n_cols,
    specs=[[{"type": "domain"}] * n_cols for _ in range(n_rows)],
    subplot_titles=[f"<b>{c}</b>" for c in cats_unicas],
    vertical_spacing=0.14,
    horizontal_spacing=0.08
)

paleta_colores = px.colors.qualitative.Prism

for idx, cat in enumerate(cats_unicas):
    r, c = (idx // n_cols) + 1, (idx % n_cols) + 1
    serie = datos_dona.get(cat)
    top = serie.nlargest(top_dona) if serie is not None else pd.Series(dtype=int)
    
    if not top.empty:
        labels_raw = top.index.tolist()
        labels_plot = [wrap_label(str(l)) for l in labels_raw]
        values = top.values.tolist()
        colores = paleta_colores[:len(labels_plot)]
        
        fig_donas.add_trace(
            go.Pie(
                labels=labels_plot,
                customdata=labels_raw,
                values=values,
                hole=0.60,
                marker=dict(colors=colores, line=dict(color="#ffffff", width=2)),
                textposition="outside",
                textinfo="label+percent",
                textfont=dict(size=10),
                hovertemplate="<b>%{customdata}</b><br>Artículos: %{value} (%{percent})<extra></extra>",
                showlegend=False
            ),
            row=r,
            col=c
        )
    else:
        fig_donas.add_trace(
            go.Pie(
                labels=["Sin datos"],
                values=[1],
                hole=0.60,
                marker=dict(colors=["#f3f4f6"]),
                textinfo="none",
                hoverinfo="none",
                showlegend=False
            ),
            row=r,
            col=c
        )

fig_donas.update_layout(
    height=430 * n_rows,
    margin=dict(t=50, b=40, l=30, r=30),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig_donas, use_container_width=True)

st.divider()

## ==========================================
## SECCIÓN 3: ETIQUETAS Y TÓPICOS MODELADOS
## ==========================================
st.subheader("3. Etiquetas vs. Tópicos")
st.write("""
Cruce entre las etiquetas  y los Tópicos. Similar al anterior queremos distinguir etiquetas que esten en mas de un Tópico.

Si las etiquetas se encuentran distribuidas en distintos Tópicos entonces estos tienen información rica para representarlas. (Lo usamos más adelante)
""")
st.caption("Celdas con valores altos marcan concordancia entre la etiqueta y el tópico; dispersiones horizontales evidencian etiquetas transversales.")

top_tags_t = st.slider("Etíquetas a mostrar:", 5, 100, 20)
top_tops_t = st.slider("Tópicos a mostrar:", 5, 50, 3)

tt_tags = df_tags_top.groupby("tag")["doc_id"].nunique().nlargest(top_tags_t).index
tt_tops = df_tags_top.groupby("topic_name")["doc_id"].nunique().nlargest(top_tops_t).index

df_hm_t = df_tags_top[(df_tags_top["tag"].isin(tt_tags)) & (df_tags_top["topic_name"].isin(tt_tops))]
if not df_hm_t.empty:
    matriz_t = pd.crosstab(df_hm_t["tag"], df_hm_t["topic_name"])
    fig_hm_t = px.imshow(matriz_t, text_auto=True, aspect="auto", color_continuous_scale="Blues")
    st.plotly_chart(fig_hm_t, use_container_width=True)

st.divider()


## ==========================================
## SECCIÓN 4: PROPUESTAS DE CONSOLIDACIÓN Y NORMALIZACIÓN
## ==========================================
st.subheader("4. Etíquetas solapadas")
st.write("""
Durante la exploracíon, se identificaron algunas etiquetas repetidas, ya sea por temas ortográficss, variaciones gramaticales, etc. Buscamos automatizar una manera de identificarlas.
""")

# --------------------------------------------------
# 1. FUSIONES AUTOMÁTICAS
# --------------------------------------------------
st.markdown("#### 1. Fusiones Recomendadas (Automáticas)")
st.write("""
Casos donde consideramos la unificación de etíquetas variantes con seguridad. Pues cumple con algunos de los umbrales de los análisis que les realizamos:
""")

df_auto_display = tags_auto.rename(columns={
    "Tag Canónico": "Canónica",
    "Frecuencia Canónico": "Freq. Canónica",
    "Tag Variante": "Variante",
    "Frecuencia Variante": "Freq. Variante",
    "Similitud Léxica": "Similitud Léxica",
    "Es Diacrítico": "Es Tilde/Acento",
    "Mismo Lema": "Mismo Lema (Singular/Plural)"
})
st.dataframe(df_auto_display, use_container_width=True, hide_index=True)

with st.expander("Nota: Explicacón de las columnas y criterio de elección del canónico"):
    st.markdown("""
    ##### ¿Qué significa cada columna?
    * **Canónica:** La recomendada para conservar.
    * **Freq. Canónica:** Cantidad de artículos donde se usa la etiqueta canónica.
    * **Variante:** La etiqueta que debería absorberse.
    * **Freq. Variante:** Cantidad de artículos donde aparece la variante.
    * **Similitud Léxica:** Coeficiente de 0 a 1 según la coincidencia de caracteres entre ambas palabras.
    * **Es Tilde/Acento:** `True` si solo difieren por un acento ortográfico (*Mexico* vs. *México*).
    * **Mismo Lema (Singular/Plural):** `True` si comparten el mismo lema (ej. lema corriendo, corrió → correr).
    * **Similitud Semántica:** Coeficiente de 0 a 1 que mide la similitud mediante *embeddings* (representación vectorial númerica de las etíquetas). Evalúa el contexto o significado, ayudandonos a diferencias etíquetas escritas parecido pero con sentidos opuestos (ej. *Periodismo* vs. *Peronismo*).

    ##### ¿Cómo se define cuál queda como canónica?
    1. **Sintaxis:** La forma sin `#` sobre la que tiene hashtag (`#Eaaf` $\\rightarrow$ `Eaaf`).
    2. **Ortografía:** La forma con tilde válida sobre la que no la lleva (`Mexico` $\\rightarrow$ `México`).
    3. **Frecuencia:** Si empatan en las reglas anteriores, la variante con más notas es la elegida.


    ##### ¿Qué condiciones debe cumplir un par de etíquetas para aparecer en esta tabla?
    Para que una etiqueta sea candidata a unificacíon automática y no genere falsos positivos, debe cumplir **al menos una** de estas condiciones:
    * Ser idénticas salvo por la presencia o ausencia de tildes (`Es Tilde/Acento = True`).
    * Compartir exactamente el mismo lema (`Mismo Lema = True`, ej. singular y plural: Violencia, Violencias).
    * Superar una **similitud léxica $\ge 0.93$** (escritura casi igual) respaldada obligatoriamente por una **similitud semántica $\ge 0.85$**.
    """)
    

# --------------------------------------------------
# 2. CASOS PARA REVISIÓN EDITORIAL
# --------------------------------------------------
st.markdown("#### 2. Casos para Revisión ")
st.write("""
Del análisis anterior tambien guardamos las etíquetas que no cumplen con las condiciones para la unificación automática con toda seguridad, pero se consideran validas para una revisión manual.""")

df_manual_display = tags_manual.rename(columns={
    "Tag Variante": "Etiqueta A",
    "Frecuencia Variante": "Uso A",
    "Tag Canónico": "Etiqueta B",
    "Frecuencia Canónico": "Uso B",
    "Es Diacrítico": "Es Tilde/Acento",
    "Mismo Lema": "Mismo Lema",
    "Similitud Léxica": "Similitud Léxica",
    "Similitud Semántica": "Similitud Semántica"
})
st.dataframe(df_manual_display, use_container_width=True, hide_index=True)

# with st.expander("Nota: Criterio por el cual no fueron consideras para la unificación automatica pero si para revisión."):
#     st.markdown(""" """)

# --------------------------------------------------
# 3. SOLAPAMIENTO TEMÁTICO VÍA TÓPICOS
# --------------------------------------------------
st.markdown("#### 3. Etiquetas Similares por Tópicos")
st.write("""
Una de las opciones que trabajamos mientras construiamos las metricas presentadas fue el utilizar los Tópicos de los artículos.
No lo terminamos usando para ello porque en general las etíquetas no significaban lo mismo. Pero, los contextos en los que se usan si son similares, entonces nos parecio interesante de mostrar.
""")

df_solap_display = tags_solapamiento.rename(columns={
    "Tag A": "Etiqueta A",
    "Tag B": "Etiqueta B",
    "Similitud_Coseno_Prob_Topicos": "Similitud"
})
st.dataframe(df_solap_display, use_container_width=True, hide_index=True)

with st.expander("Metodología y Métricas"):
    st.markdown("""
    * **Vectores de tópicos:** Cada artículo tiene un vector que indica qué proporción le corresponde de cada tópico del modelo.
    * **Vector promedio de la etiqueta:** Promediamos los vectores de todas los artículos que llevan una etiqueta determinada, quedando un vector promedio de los tópicos para esa etíqueta.
    * **Similitud coseno:** Calculamos la correlación entre los perfiles de cada par de etiquetas. Mostramos pares con similitud $\ge 0.85$.
    * **Interpretación:** Un valor alto indica que ambas etiquetas se aplican sobre artículos con contextos semánticos similares, puramente ortográfico no llega a ver.
    """)

st.write("")
st.write("Buscar Similares a una etíqueta especifica")
# Universo de etiquetas que tienen al menos una relación de solapamiento
tags_con_similares = sorted(
    set(df_solap_display["Etiqueta A"].dropna()).union(set(df_solap_display["Etiqueta B"].dropna()))
)

tag_especifica = st.selectbox(
    "Seleccioná una etiqueta:",
    options=tags_con_similares,
    index=495 if tags_con_similares else None,
    placeholder="Escribí para buscar...",
    label_visibility="collapsed"
)

if tag_especifica:
    # Filtramos donde aparezca en A o en B
    mask = (df_solap_display["Etiqueta A"] == tag_especifica) | (df_solap_display["Etiqueta B"] == tag_especifica)
    df_filtrado = df_solap_display[mask].copy()

    # Mapeamos para quedarnos con la contraparte y su similitud
    relacionadas = []
    for _, row in df_filtrado.iterrows():
        etiqueta_vecina = row["Etiqueta B"] if row["Etiqueta A"] == tag_especifica else row["Etiqueta A"]
        relacionadas.append({
            "Etiqueta Similar": etiqueta_vecina,
            "Similitud Temática": row["Similitud"]
        })

    df_vecinas = pd.DataFrame(relacionadas).sort_values(by="Similitud Temática", ascending=False).reset_index(drop=True)

    st.dataframe(
        df_vecinas,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Etiqueta Similar": st.column_config.TextColumn("Etiquetas Similares", width="large"),
            "Similitud Temática": st.column_config.ProgressColumn(
                "Similitud",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
                width="medium"
            ),
        }
    )
# # Buscamos coincidencias aproximadas
# coincidencias = [
#     (i, tag) for i, tag in enumerate(tags_con_similares) 
#     if "messi" in tag.lower()
# ]

# st.write("Coincidencias encontradas (Índice, Nombre):", coincidencias)