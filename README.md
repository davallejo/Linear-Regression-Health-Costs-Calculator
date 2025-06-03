# 💰 Linear Regression Health Costs Calculator

Este proyecto tiene como objetivo predecir los costos de atención médica a partir de un conjunto de datos reales, utilizando un modelo de **Regresión Lineal** implementado con **TensorFlow/Keras**.

---

## 🎯 Objetivo del Proyecto

Construir un modelo de regresión que prediga el costo de los gastos médicos de una persona en función de atributos como edad, IMC, número de hijos, sexo, tabaquismo y región. El modelo debe alcanzar una **error absoluto medio (MAE)** menor a **$3,500** para aprobar el desafío.

---

## 📖 Introducción

El conjunto de datos utilizado proviene de FreeCodeCamp y contiene información sobre 1,338 personas, con las siguientes variables:

- `age`: Edad del paciente.
- `sex`: Sexo.
- `bmi`: Índice de masa corporal.
- `children`: Número de hijos dependientes.
- `smoker`: Si la persona fuma.
- `region`: Región geográfica.
- `expenses`: Costos médicos (variable objetivo).

---

## ⚙️ Tecnología Usada

- 🐍 Python 3.x
- 📊 Pandas, NumPy, Matplotlib
- 🤖 Scikit-learn
- 🔬 TensorFlow 2.x y Keras
- 📁 Google Colab
- 🧪 TensorFlow Docs para visualización

---

## 🧪 Proceso de Desarrollo

### 1. 📥 Carga y Visualización del Dataset

Se descargó el archivo `insurance.csv` desde el repositorio de FreeCodeCamp.

```python
dataset = pd.read_csv('insurance.csv')
```

### 2. 🔄 Preprocesamiento
- Normalización de variables numéricas (age, bmi, children) con StandardScaler.
- Codificación One-Hot para variables categóricas (sex, smoker, region) usando OneHotEncoder.
- Concatenación de datos procesados en un nuevo DataFrame.
- Separación en train_dataset y test_dataset (80% / 20%).

### 3. 🧠 Construcción del Modelo
Modelo Sequential con arquitectura:
```python
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=[num_features]),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(1)
])
```
- Optimización: Adam con learning_rate=0.001
- Pérdida: mean_squared_error
- Métricas: mae, mse
- Validación: 20% del conjunto de entrenamiento
- EarlyStopping aplicado con patience=10

### 4. 📊 Evaluación del Modelo
Se utilizó el conjunto test_dataset para evaluar el rendimiento final del modelo:
```python
loss, mae, mse = model.evaluate(test_dataset, test_labels)
```

## ✅ Resultados Obtenidos
- MAE final: 3243.73 dólares
- Condición cumplida: MAE < 3500
- Mensaje: You passed the challenge. Great job!
Además, se generó una gráfica de dispersión entre los valores reales y las predicciones del modelo.
![image](https://github.com/user-attachments/assets/a2c619b4-c8e6-4b86-ad9a-53c1af024b61)

## 🧾 Conclusión
Este proyecto demuestra cómo aplicar técnicas de regresión y preprocesamiento de datos para construir un modelo predictivo eficiente. Se logró un desempeño óptimo al combinar normalización, codificación categórica, y regularización con capas Dropout. El modelo puede ser utilizado como base para sistemas de predicción de costos médicos más complejos.

