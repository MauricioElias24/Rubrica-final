# Iris Species Classification - Proyecto Final de Mineria de Datos

## Descripcion del Proyecto

Este proyecto consiste en un clasificador de especies de flores Iris utilizando el algoritmo **Random Forest**. 
Se desarrollo un dashboard interactivo con **Streamlit** que permite:

- Visualizar metricas de rendimiento del modelo
- Ingresar medidas de una flor y predecir su especie
- Ver un grafico 3D interactivo donde se muestra la nueva muestra junto al dataset
- Explorar visualizaciones adicionales como histogramas y matriz de dispersion

## Dataset: Iris Flower

El dataset Iris es uno de los mas conocidos en machine learning. Contiene **150 muestras** de flores Iris,
cada una con **4 caracteristicas numericas**:

| Caracteristica | Descripcion | Rango tipico |
|---------------|-------------|--------------|
| Largo del Sepalo | Longitud del sepalo en cm | 4.3 - 7.9 |
| Ancho del Sepalo | Ancho del sepalo en cm | 2.0 - 4.4 |
| Largo del Petalo | Longitud del petalo en cm | 1.0 - 6.9 |
| Ancho del Petalo | Ancho del petalo en cm | 0.1 - 2.5 |

Y **3 especies** a clasificar:

- **Setosa** (0) — Facil de diferenciar, petalos pequeños
- **Versicolor** (1) — Intermedia
- **Virginica** (2) — Petalos mas grandes

## Metodologia (Workflow)

El proceso que segui para este proyecto fue el siguiente:

### 1. Entendimiento de los datos
Cargue el dataset con `load_iris()` de sklearn y explore las caracteristicas.
Analice estadisticas basicas (media, desviacion, etc.) y visualice la distribucion
de cada caracteristica por especie para entender como se diferencian.

### 2. Preprocesamiento
- Separe las caracteristicas (X) de la variable objetivo (y)
- Dividi los datos en 70% para entrenamiento y 30% para prueba usando `train_test_split`
- No fue necesario escalar los datos porque Random Forest no lo requiere

### 3. Modelado
Eleji **Random Forest Classifier** por las siguientes razones:
- Funciona bien con datasets pequeños como este (150 muestras)
- No necesita escalado ni normalizacion de los datos
- Maneja bien relaciones no lineales entre las caracteristicas
- Proporciona probabilidades para cada clase, no solo la prediccion
- Es resistente al overfitting gracias al promedio de multiples arboles

Configure el modelo con 100 arboles (`n_estimators=100`) y una semilla aleatoria
(`random_state=42`) para que los resultados sean reproducibles.

### 4. Evaluacion
Para medir el rendimiento del modelo use las siguientes metricas:

| Metrica | Que mide |
|---------|----------|
| **Accuracy** | Porcentaje total de predicciones correctas |
| **Precision** | De las que el modelo dijo que eran de una especie, cuantas realmente lo eran |
| **Recall** | De las que realmente eran de una especie, cuantas identifico correctamente |
| **F1-Score** | Media armonica entre Precision y Recall |

### 5. Visualizacion
Cree un dashboard interactivo con Streamlit que incluye:
- Tabla de metricas del modelo
- Panel de entrada con sliders para las 4 medidas
- Prediccion en tiempo real con la especie y probabilidades
- Grafico 3D interactivo (largo sepalo, ancho sepalo, largo petalo)
- Histogramas de distribucion por especie
- Matriz de dispersion de todas las caracteristicas

## Requisitos

- Python 3.8 o superior
- Las librerias listadas en `requirements.txt`


## Archivos incluidos

| Archivo | Descripcion |
|---------|-------------|
| `Proyect.py` | Aplicacion principal de Streamlit |
| `Iris_Classification_Colab.ipynb` | Notebook para Google Colab (exploracion y entrenamiento) |
| `requirements.txt` | Lista de dependencias |
| `README.md` | Este archivo |

## Integrantes

Mauricio Bustillo 

Iris Giraldo

## Enlace al video

https://drive.google.com/drive/folders/1FSd2RKA-w7gJIXGYgUI-yT2sV_Uys9Ex?usp=sharing

---

**Universidad de la Costa**
**Mineria de Datos - 2026**
