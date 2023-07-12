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
    | Nombre | Participación|
    |--|--|
    | Roger Cabrera | Support Vector Machine, Long Short-Term Memory |

    ### Especificaciones:
    **Donde muestra las predicciones/los resultados:**
    - Gráficamente. 
    - Númericamente los valores de las predicciones (print de dataframe con la predicción o clasificación).
    

    **Donde el usuario pueda indicar:**
    - El modelo ejecutar.
    - La acción o instrumento financiero que quiera analizar.
"""
)