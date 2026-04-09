# Explicacion del Funcionamiento del Sistema

Este documento resume como funciona el sistema de seleccion de proveedores para poder explicarlo en una exposicion academica o tecnica.

## 1. Que problema resuelve

El sistema ayuda a decidir si un proveedor debe ser seleccionado o no, usando datos historicos y un modelo de aprendizaje supervisado (arbol de decision).

## 2. Flujo general del sistema

1. Carga de datos (CSV de ejemplo o CSV propio).
2. Seleccion de variable objetivo y variables predictoras.
3. Entrenamiento del modelo de arbol de decision.
4. Evaluacion del rendimiento del modelo.
5. Interpretacion de resultados (importancia y reglas).
6. Prediccion de nuevos proveedores.

## 3. Modulo de interfaz (Streamlit)

La aplicacion principal esta en `app.py` y cumple estas funciones:

- Permite elegir la fuente de datos.
- Muestra vista previa del dataset.
- Deja configurar hiperparametros del arbol:
  - criterio (`gini`, `entropy`, `log_loss`),
  - profundidad maxima,
  - minimo de muestras para dividir,
  - porcentaje de prueba (`test_size`).
- Dispara el entrenamiento con el boton "Entrenar modelo".
- Presenta metricas y reportes.
- Permite hacer prediccion de un nuevo proveedor mediante formulario.

## 4. Modulo de inteligencia (pipeline)

La logica del modelo esta en `src/pipeline.py`:

### 4.1 Separacion de datos

`split_features_target` separa:

- X: variables de entrada.
- y: variable objetivo (por ejemplo, `Seleccionado`).

### 4.2 Preprocesamiento automatico

`train_decision_tree` construye un `Pipeline` con `ColumnTransformer`:

- Columnas numericas:
  - imputacion de faltantes con mediana.
- Columnas categoricas:
  - imputacion con valor mas frecuente,
  - codificacion One-Hot.

Esto permite entrenar aunque existan datos faltantes o texto en columnas.

### 4.3 Entrenamiento del arbol de decision

El clasificador usado es `DecisionTreeClassifier` de scikit-learn.

Parametros principales:

- `criterion`: funcion de division del arbol.
- `max_depth`: limita complejidad para evitar sobreajuste.
- `min_samples_split`: controla cuando se puede dividir un nodo.

Se realiza division entrenamiento/prueba con `train_test_split` y estratificacion cuando aplica.

## 5. Evaluacion del modelo

`evaluate_model` calcula:

- Accuracy.
- Precision (weighted).
- Recall (weighted).
- F1-score (weighted).
- Matriz de confusion.
- Reporte de clasificacion por clase.

Estas salidas permiten validar si el modelo es confiable para apoyar decisiones.

## 6. Interpretabilidad

El sistema incluye dos salidas clave para justificar resultados:

1. `feature_importances`: ordena variables por impacto en la decision.
2. `decision_rules`: exporta reglas del arbol en texto.

Con esto, se puede explicar no solo que decide el modelo, sino por que lo decide.

## 7. Prediccion de un nuevo proveedor

Tras entrenar:

1. El usuario completa un formulario con valores de las variables predictoras.
2. El pipeline aplica exactamente el mismo preprocesamiento.
3. El modelo devuelve la clase predicha.
4. Si esta disponible, tambien muestra probabilidades por clase.

## 8. Valor del sistema para negocio

- Estandariza criterios de seleccion de proveedores.
- Reduce subjetividad en la decision.
- Aporta trazabilidad por medio de reglas e importancia de variables.
- Permite simulacion rapida de escenarios (cambiar variables y predecir).

## 9. Guion corto para exposicion (1-2 minutos)

"Este sistema toma datos historicos de proveedores y entrena un arbol de decision para clasificar si un proveedor debe ser seleccionado. Primero cargamos el dataset, definimos objetivo y variables, y entrenamos el modelo con preprocesamiento automatico para datos numericos y categoricos. Luego evaluamos su desempeno con accuracy, precision, recall, f1 y matriz de confusion. Finalmente interpretamos el resultado mediante importancia de variables y reglas del arbol, y hacemos predicciones de nuevos proveedores en tiempo real desde la interfaz Streamlit."
