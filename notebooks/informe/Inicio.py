import streamlit as st

st.set_page_config(
    page_title="Anfibia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y bajada editorial
st.title("Informe - Proyecto Anfibia")

st.markdown("""Este informe reúne los avances de trabajo sobre la base de datos de la revista. A partir de técnicas de procesamiento automático del lenguaje, analizamos los artículos, identificamos "Tópicos" según los mismos y construimos prototipos para organizar el etiquetado y recomendar artículos similares.""")

st.divider()

# Estructura del informe
st.markdown("### Estructura del informe")
st.markdown("Navegá desde el menú lateral para explorar estos cuatro puntos:")

# Fila 1: Puntos 1 y 2
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### 1. Análisis de los artículos")
        st.write("Una exploración breve por el vocabulario de los artículos de la base de datos, sus categorías y cómo cambiaron año a año.")

with col2:
    with st.container(border=True):
        st.markdown("#### 2. Creación y Exploración de Tópicos")
        st.write("Mediante técnicas de Procesamiento del Lenguaje Natural (Modelado de Tópicos), se agruparon de manera no supervisada 'Tópicos' y se examinan sus significados.")

# Fila 2: Puntos 3 y 4
col3, col4 = st.columns(2)

with col3:
    with st.container(border=True):
        st.markdown("#### 3. Examinación de las Etiquetas")
        st.write("Un diagnóstico a fondo sobre las etiquetas: Su distribución, etiquetas duplicadas y sugerencias.")

with col4:
    with st.container(border=True):
        st.markdown("#### 4. Recomendación de Artículos")
        st.write("Tomando provecho de la información de los tópicos de cada artículo, proponemos un sistema de recomendación en base a estos.")

st.divider()

st.caption("Equipo: Natalia Debandi (CIAI/UNSAM)  ·  Manuel Szewc (CIAI/UNSAM)  ·  Mauricio Kreszczuk (Estudiante LCD UNSAM)")