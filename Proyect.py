import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuracion de la pagina
st.set_page_config(page_title="Clasificador de Iris", layout="wide")

st.title("Clasificacion de Especies Iris")
st.write("Proyecto final de Mineria de Datos")

# Cargar el dataset iris
iris = load_iris()
datos = iris.data
etiquetas = iris.target

# Crear un DataFrame con los datos
df = pd.DataFrame(datos, columns=iris.feature_names)
df['especie'] = etiquetas

# Renombrar las columnas para que sea mas facil de entender
df.columns = ['largo_sepalo', 'ancho_sepalo', 'largo_petalo', 'ancho_petalo', 'especie']

# Mapear los numeros a nombres de especies
especies_nombres = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
df['especie_nombre'] = df['especie'].map(especies_nombres)

# Separar caracteristicas (X) y target (y)
X = df[['largo_sepalo', 'ancho_sepalo', 'largo_petalo', 'ancho_petalo']]
y = df['especie']

# Dividir en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Hacer predicciones con los datos de prueba
y_pred = modelo.predict(X_test)

# Calcular metricas de rendimiento
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# ============== SIDEBAR: ENTRADA DE DATOS ==============
st.sidebar.header("Ingresa las medidas de la flor")

largo_sepalo = st.sidebar.slider("Largo del Sepalo (cm)", 4.0, 8.0, 5.8)
ancho_sepalo = st.sidebar.slider("Ancho del Sepalo (cm)", 2.0, 4.5, 3.0)
largo_petalo = st.sidebar.slider("Largo del Petalo (cm)", 1.0, 7.0, 3.5)
ancho_petalo = st.sidebar.slider("Ancho del Petalo (cm)", 0.1, 2.5, 1.2)

# Crear un DataFrame con los valores que ingreso el usuario
datos_usuario = pd.DataFrame({
    'largo_sepalo': [largo_sepalo],
    'ancho_sepalo': [ancho_sepalo],
    'largo_petalo': [largo_petalo],
    'ancho_petalo': [ancho_petalo]
})

# Predecir la especie con el modelo
prediccion = modelo.predict(datos_usuario)[0]
probabilidad = modelo.predict_proba(datos_usuario)[0]

# ============== COLUMNA 1: METRICAS ==============
col1, col2 = st.columns(2)

with col1:
    st.subheader("Metricas del Modelo")
    
    st.write(f"**Accuracy:** {accuracy:.3f}")
    st.write(f"**Precision:** {precision:.3f}")
    st.write(f"**Recall:** {recall:.3f}")
    st.write(f"**F1-Score:** {f1:.3f}")
    
    st.write("")
    st.write("El modelo tuvo un rendimiento de ", accuracy*100, "% de accuracy")

# ============== COLUMNA 2: PREDICCION ==============
with col2:
    st.subheader("Prediccion")
    
    especie_predicha = especies_nombres[prediccion]
    st.markdown(f"### {especie_predicha}")
    
    # Grafico de barras con las probabilidades
    prob_df = pd.DataFrame({
        'Especie': ['Setosa', 'Versicolor', 'Virginica'],
        'Probabilidad': probabilidad
    })
    st.bar_chart(prob_df.set_index('Especie'))

# ============== GRAFICO 3D ==============
st.subheader("Grafico 3D: Muestra nueva vs Dataset")

# Preparar los datos para el grafico
df_viz = df.copy()

# Crear el grafico 3D
figura_3d = px.scatter_3d(
    df_viz,
    x='largo_sepalo',
    y='ancho_sepalo',
    z='largo_petalo',
    color='especie_nombre',
    color_discrete_map={
        'Setosa': 'blue',
        'Versicolor': 'red',
        'Virginica': 'green'
    },
    title="Visualizacion 3D del dataset Iris",
    labels={
        'largo_sepalo': 'Largo Sepalo (cm)',
        'ancho_sepalo': 'Ancho Sepalo (cm)',
        'largo_petalo': 'Largo Petalo (cm)'
    }
)

# Agregar el punto de la muestra nueva
figura_3d.add_trace(
    go.Scatter3d(
        x=[largo_sepalo],
        y=[ancho_sepalo],
        z=[largo_petalo],
        mode='markers',
        marker=dict(
            size=12,
            color='black',
            symbol='x'
        ),
        name='Mi muestra'
    )
)

st.plotly_chart(figura_3d, use_container_width=True)

# ============== GRAFICOS ADICIONALES ==============
st.subheader("Visualizaciones adicionales")

tab1, tab2 = st.tabs(["Histogramas", "Matriz de Dispersion"])

with tab1:
    # Histogramas de cada caracteristica separada por especie
    fig_hist = px.histogram(
        df_viz,
        x='largo_sepalo',
        color='especie_nombre',
        nbins=20,
        title="Distribucion del Largo del Sepalo",
        color_discrete_map={
            'Setosa': 'blue',
            'Versicolor': 'red',
            'Virginica': 'green'
        }
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    fig_hist2 = px.histogram(
        df_viz,
        x='largo_petalo',
        color='especie_nombre',
        nbins=20,
        title="Distribucion del Largo del Petalo",
        color_discrete_map={
            'Setosa': 'blue',
            'Versicolor': 'red',
            'Virginica': 'green'
        }
    )
    st.plotly_chart(fig_hist2, use_container_width=True)

with tab2:
    # Matriz de dispersion
    fig_matrix = px.scatter_matrix(
        df_viz,
        dimensions=['largo_sepalo', 'ancho_sepalo', 'largo_petalo', 'ancho_petalo'],
        color='especie_nombre',
        color_discrete_map={
            'Setosa': 'blue',
            'Versicolor': 'red',
            'Virginica': 'green'
        },
        title="Matriz de Dispersion de las caracteristicas"
    )
    fig_matrix.update_layout(height=600)
    st.plotly_chart(fig_matrix, use_container_width=True)

st.markdown("---")
st.markdown("**Integrantes:**Mauricio Bustillo, Iris Giraldo**")
st.markdown("**Universidad de la Costa - Mineria de Datos**")
