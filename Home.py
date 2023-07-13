import streamlit as st

st.set_page_config(
    page_title=" Equipo E - Trabajo Final",
    page_icon="🤖",
)

st.write("# Aplicación Web de Modelos de Machine Learning - Equipo E")

st.sidebar.success("Seleccione un modelo del menú")

st.markdown(
    """
    # Grupo E - Integrantes:
    | Nombre | Código | Participación |
    |--|--|--|
    | Roger Omar Cabrera Silva | 19200070 | Decision Tree, Linear Regression,  |
    | Fredi Alexander Caballero Leon| 18200076 | K-Nearest Neighbours, Long Short-Term Memory, Support Vector Classifier|
    | Leonardo Chavez Calderón de la Barca| 19200076 | Random Forest, Support Vector Classifier|

    ### Características:
    - Gráficos con su interpretación. 
    - Valores de las Predicciones
    - Ploteo de Precios Reales
    - 4 modelos diferentes a usar
    - Cuadro de texto para ingresar la acción
    - Abarca todo el registro histórico de la acción
"""
)