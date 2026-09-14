import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

st.set_page_config(page_title="Recomendador", layout="wide")

# ==========================================
# RUTAS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "streamlit"

# ==========================================
# CARGA DE DATOS CACHEADA
# ==========================================
@st.cache_data
def load_data():
    df_rec = pd.read_csv(DATA_DIR / "df_recomendador_base.csv")
    df_rec["post_date"] = pd.to_datetime(df_rec["post_date"], errors="coerce")
    
    # Cargamos el diccionario para extraer las URLs reales
    df_titulos = pd.read_csv(DATA_DIR / "diccionario_titulos_url.csv")
    df_titulos["ID"] = df_titulos["ID"].astype(int)
    df_rec["doc_id"] = df_rec["doc_id"].astype(int)
    
    df_rec["source_url"] = df_titulos['source_url']
    
    # # Fallback por si algún registro no tuviera coincidencia
    # df_rec["source_url"] = df_rec["source_url"].fillna("https://revistaanfibia.com/?p=" + df_rec["doc_id"].astype(str))
    
    matriz_dist = np.load(DATA_DIR / "doc_topic_dist.npy")
    return df_rec, matriz_dist

# Ejecución y asignación obligatoria de la carga
df_rec, matriz_dist = load_data()

# ==========================================
# VISTA Y LÓGICA
# ==========================================
st.subheader("Motor de Recomendación")
st.write("""
Buscábamos encontrar una forma de detectar artículos similares en base a los textos de los mismos. Los tópicos que entrenamos ya contienen esa información. Así que, teniendo cada artículo una distribución de tópicos, para un artículo dado buscamos otros que tengan una distribución similar.
""")

with st.expander("Metodología: ¿Cómo calcula la similitud?"):
    st.markdown("""
    * **Similitud Coseno sobre Tópicos:** Se toma el vector de distribución de tópicos del artículo original y se mide la similitud coseno contra todos los demás artículos del archivo. Valores cercanos a 1.0 indican artículos casi idénticos.
    * **Proyección UMAP:** Como es imposible graficar vectores de múltiples dimensiones, usamos UMAP para comprimir los datos a 2D.
    """)

opciones_articulos = df_rec["doc_id"].astype(str) + " - " + df_rec["title_text"]
articulo_sel = st.selectbox("Buscar artículo por ID o Título:", opciones_articulos)
target_id = int(articulo_sel.split(" - ")[0])

# Datos del artículo elegido
target_row = df_rec[df_rec["doc_id"] == target_id].iloc[0]

def graficar_constelacion(target_id, recomendados_ids, df_completo, col_x, col_y, titulo, opacidad_fondo=0.45):
    df_resto = df_completo[~df_completo["doc_id"].isin([target_id] + recomendados_ids)]
    df_rec_local = df_completo.loc[df_completo["doc_id"].isin(recomendados_ids)].copy()
    df_target = df_completo[df_completo["doc_id"] == target_id]
    
    # 1. FONDO (WebGL)
    fig = px.scatter(
        df_resto, 
        x=col_x, 
        y=col_y, 
        color='topic_label', 
        color_discrete_sequence=px.colors.qualitative.Alphabet, 
        custom_data=['topic_label', 'title_text', 'tagline'],
        render_mode="webgl" 
    )
    fig.update_traces(
        marker=dict(size=5, opacity=opacidad_fondo), 
        hovertemplate="<b>%{customdata[0]}</b><br><b>%{customdata[1]}</b><br><i>%{customdata[2]}</i><extra></extra>"
    )
    
    # 2. CAPA VISUAL (WebGL)
    fig.add_trace(go.Scattergl(
        x=df_rec_local[col_x], 
        y=df_rec_local[col_y], 
        mode="markers", 
        name="Recomendados", 
        marker=dict(symbol="circle", size=14, line=dict(width=2, color="#1f77b4")),
        hoverinfo="skip"
    ))
    
    fig.add_trace(go.Scattergl(
        x=df_target[col_x], 
        y=df_target[col_y], 
        mode="markers", 
        name="Original", 
        marker=dict(color="#d62728", size=18, symbol="diamond"), 
        hoverinfo="skip"
    ))
    
    # 3. CAPA FANTASMA (SVG)
    fig.add_trace(go.Scatter(
        x=df_rec_local[col_x], 
        y=df_rec_local[col_y], 
        mode="markers", 
        showlegend=False, 
        marker=dict(symbol="circle", size=18, color="rgba(0,0,0,0)"), 
        customdata=df_rec_local[["title_text", "topic_label", "tagline"]], 
        hovertemplate="<b>Recomendación</b><br><b>Tópico:</b> %{customdata[1]}<br><b>%{customdata[0]}</b><br><i>%{customdata[2]}</i><extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=df_target[col_x], 
        y=df_target[col_y], 
        mode="markers", 
        showlegend=False, 
        marker=dict(color="rgba(0,0,0,0)", size=18, symbol="diamond", line=dict(width=2, color="black")), 
        customdata=df_target[["title_text", "topic_label", "tagline"]], 
        hovertemplate="<b>ORIGINAL</b><br><b>Tópico:</b> %{customdata[1]}<br><b>%{customdata[0]}</b><br><i>%{customdata[2]}</i><extra></extra>"
    ))
    
    fig.update_layout(
        title=f"<b>{titulo}</b>", 
        height=700, 
        xaxis=dict(showticklabels=False), 
        yaxis=dict(showticklabels=False),
        hovermode="closest"
    )
    return fig
    
def graficar_temporal(target_id, similitudes_array, top_n=100):
    df_temp = df_rec.copy()
    df_temp["Similitud"] = similitudes_array
    
    df_origen = df_temp[df_temp["doc_id"] == target_id]
    df_recs = df_temp[df_temp["doc_id"] != target_id].sort_values("Similitud", ascending=False).head(top_n).copy()
    df_recs["Rank"] = range(1, top_n + 1)
    
    df_top10 = df_recs[df_recs["Rank"] <= 10]
    df_resto = df_recs[df_recs["Rank"] > 10]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_resto["post_date"], 
        y=df_resto["Similitud"], 
        mode="markers", 
        name="Rank 11-100", 
        marker=dict(color="#ab63fa", size=8, opacity=0.5),
        customdata=df_resto[["title_text", "Rank", "topic_label"]], 
        hovertemplate="<b>Rank #%{customdata[1]}</b><br>%{customdata[0]}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df_top10["post_date"], 
        y=df_top10["Similitud"], 
        mode="markers", 
        name="Top 10", 
        marker=dict(symbol="circle", color="#FFD700", size=16, line=dict(width=2, color="black"), opacity=1.0), 
        customdata=df_top10[["title_text", "Rank", "topic_label"]], 
        hovertemplate="<b>Rank #%{customdata[1]}</b><br>%{customdata[0]}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df_origen["post_date"], 
        y=df_origen["Similitud"], 
        mode="markers", 
        name="Original", 
        marker=dict(symbol="diamond", color="#d62728", size=18, line=dict(width=2, color="black"), opacity=1.0), 
        customdata=df_origen[["title_text", "topic_label"]], 
        hovertemplate="<b>ORIGINAL</b><br>%{customdata[0]}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Distribución Temporal de Similitud", 
        height=500,
        hovermode="closest"
    )
    return fig

#######################
########## SECCIONES
#######################

if target_id is not None:
    url_articulo = target_row.get("source_url")
    if pd.isna(url_articulo) or not isinstance(url_articulo, str) or not url_articulo.startswith("http"):
        url_articulo = f"https://revistaanfibia.com/?p={target_id}"

    #st.markdown(f"**Artículo seleccionado:** {target_row['title_text']}")
    if pd.notna(target_row['tagline']) and str(target_row['tagline']).strip():
        st.caption(f"_{target_row['tagline']}_")
    st.link_button("Ver artículo original ↗", url_articulo)
    st.divider()

    ############ TOP 10 ###########################
    target_idx = df_rec[df_rec["doc_id"] == target_id].index[0]
    target_vector = matriz_dist[target_idx].reshape(1, -1)
    sims = cosine_similarity(target_vector, matriz_dist).flatten()
    
    indices_top = sims.argsort()[::-1][:11]
    indices_top = [i for i in indices_top if i != target_idx][:10]
    recs_ids = df_rec.iloc[indices_top]["doc_id"].tolist()
    
    st.write("### Top 10 Recomendaciones")
    st.caption("Los artículos con mayor similitud a la nota original.")
    
    df_mostrar = df_rec.iloc[indices_top][["doc_id", "title_text", "topic_label", "post_date", "source_url"]].copy()
    df_mostrar["Similitud"] = sims[indices_top].round(4)
    
    df_mostrar = df_mostrar.rename(columns={
        "doc_id": "ID",
        "title_text": "Título",
        "topic_label": "Tópico",
        "post_date": "Publicación",
        "source_url": "Enlace"
    })
    
    st.dataframe(
        df_mostrar,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("ID", format="%d", width="small", alignment="center"),
            "Título": st.column_config.TextColumn("Título", width="large"),
            "Tópico": st.column_config.TextColumn("Tópico", width="small"),
            "Publicación": st.column_config.DatetimeColumn("Publicación", format="YYYY-MM-DD", width="small", alignment="center"),
            "Similitud": st.column_config.ProgressColumn("Similitud", format="%.4f", min_value=0.0, max_value=1.0, width="small"),
            "Enlace": st.column_config.LinkColumn("Enlace", display_text="Ver", width="small", alignment="center")
        }
    )

    ##### MAPAS DE TOPICOS ########################################

    st.markdown("#### Mapas de Tópicos")
    st.write("Visualización espacial de los artículos, para ver qué tan aislado o cercano a las recomendaciones está el artículo original.")
    
    val_opacidad = st.slider(
        "Ajustar opacidad de los artículos aparte del seleccionado y sus recomendaciones", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.35, 
        step=0.05,
        help="Bajá la opacidad si el enjambre de puntos te impide distinguir los artículos recomendados."
    )
    st.caption("Al pasar el cursor sobre los puntos, se muestran su nombre y Tagline." \
               "  \nSeleccioná un área para hacer zoom y hacé doble clic sobre el gráfico para reestablecer la vista. Usá la leyenda de la derecha para encender, apagar o aislar Tópicos.")
    st.caption("**1. Mapa Semántico:** La ubicación y agrupación de los artículos está basada en el vocabulario y las estructuras de los textos (Embeddings)." \
    "  \nPermite ver qué tan similares son los textos de los artículos recomendados.")
    st.plotly_chart(graficar_constelacion(target_id, recs_ids, df_rec, "umap_sem_x", "umap_sem_y", "Mapa Semántico", val_opacidad), use_container_width=True)
    
    st.caption("**2. Mapa Tópico:** La ubicación y agrupación de los artículos se basa en la distribución de los Tópicos para cada artículo." \
    "  \nComo los puntos están agrupados por los Tópicos, permite ver rápidamente si las recomendaciones pertenecen a diferentes Tópicos.")
    st.plotly_chart(graficar_constelacion(target_id, recs_ids, df_rec, "umap_tem_x", "umap_tem_y", "Mapa Tópico", val_opacidad), use_container_width=True)


    ############ TIEMPO ######################
    st.markdown("#### Distribución Temporal")
    st.write("Muestra en que momento se públicaron los artículos que estamos recomendando. Queriamos analizar si las recomendaciónes estan sesgadas por el tiempo.")
    
    st.plotly_chart(graficar_temporal(target_id, sims), use_container_width=True)