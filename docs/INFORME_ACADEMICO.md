# Informe Academico

## Titulo

Desarrollo e implementacion de un sistema de seleccion de proveedores basado en arboles de decision con Python y Streamlit

## Resumen

Este proyecto implementa un sistema de apoyo a decisiones para la seleccion de proveedores usando tecnicas de aprendizaje supervisado. La solucion integra una interfaz web en Streamlit y un modelo de arbol de decision en scikit-learn para clasificar proveedores como seleccionados o no seleccionados a partir de variables operativas y de desempeno. El sistema incluye preprocesamiento automatico, entrenamiento configurable, evaluacion mediante metricas estandar e interpretabilidad por reglas del arbol e importancia de variables. Los resultados permiten reducir la subjetividad en el proceso de seleccion y mejorar la trazabilidad de la decision.

## 1. Introduccion

La gestion de proveedores es una actividad critica para organizaciones que dependen de compras y abastecimiento continuo. En muchos entornos, la seleccion de proveedores se realiza con criterios parcialmente subjetivos, lo que puede afectar calidad, costos y cumplimiento logistico. En este contexto, la analitica de datos permite transformar historicos de desempeno en reglas de decision objetivas.

Este trabajo propone un sistema orientado a clasificar proveedores mediante un arbol de decision, debido a su equilibrio entre capacidad predictiva e interpretabilidad. A diferencia de modelos caja negra, el arbol facilita justificar cada resultado ante responsables de compras y auditoria.

## 2. Planteamiento del problema

Problema principal:

- No existe un mecanismo estandarizado y explicable para decidir la seleccion de proveedores basado en evidencia historica.

Preguntas orientadoras:

- Que variables explican mejor la seleccion de proveedores?
- Que tan confiable es un modelo de clasificacion para apoyar esta decision?
- Como presentar resultados de forma comprensible para usuarios no tecnicos?

## 3. Objetivos

### 3.1 Objetivo general

Desarrollar e implementar un sistema de seleccion de proveedores basado en arboles de decision con Python y Streamlit.

### 3.2 Objetivos especificos

- Construir un flujo de carga y validacion de datos de proveedores.
- Entrenar un modelo de clasificacion con preprocesamiento de variables numericas y categoricas.
- Evaluar el desempeno del modelo con metricas de clasificacion.
- Incorporar mecanismos de interpretabilidad para justificar decisiones.
- Implementar una interfaz interactiva para entrenamiento y prediccion.

## 4. Marco metodologico

### 4.1 Enfoque

El enfoque es aplicado y cuantitativo, con desarrollo de software y experimentacion sobre datos tabulares.

### 4.2 Datos

El sistema trabaja con archivos CSV con una variable objetivo binaria o multiclase (por ejemplo, Seleccionado) y un conjunto de variables explicativas como precio, calidad, cumplimiento y ubicacion.

### 4.3 Pipeline de modelado

Se implementa un pipeline con las siguientes etapas:

1. Separacion de variables X e y.
2. Deteccion automatica de variables numericas y categoricas.
3. Imputacion de faltantes:
   - numericas con mediana,
   - categoricas con moda.
4. Codificacion one-hot de categoricas.
5. Entrenamiento de DecisionTreeClassifier.
6. Evaluacion sobre conjunto de prueba.

### 4.4 Herramientas

- Python 3
- Streamlit
- Pandas
- Scikit-learn

## 5. Arquitectura de la solucion

Componentes principales:

- app.py: capa de presentacion e interaccion con usuario.
- src/pipeline.py: capa de logica de negocio y machine learning.
- data/proveedores_ejemplo.csv: dataset de referencia para pruebas.

Flujo de uso:

1. El usuario define fuente de datos y parametros.
2. El sistema entrena y evalua el modelo.
3. Se muestran metricas, tablas y reglas.
4. El usuario ingresa un nuevo proveedor y obtiene prediccion.

## 6. Resultados y analisis

El sistema entrega de forma automatizada:

- Accuracy del modelo.
- Precision, recall y F1 ponderados.
- Matriz de confusion.
- Reporte de clasificacion por clase.
- Importancia de variables.
- Reglas del arbol para interpretacion.

Interpretacion:

- Las metricas globales permiten valorar confiabilidad.
- La matriz de confusion identifica tipos de error.
- Las importancias y reglas permiten explicar por que se selecciona o rechaza un proveedor.

## 7. Aportes

- Estandariza criterios de seleccion de proveedores.
- Disminuye variabilidad entre evaluadores humanos.
- Mejora transparencia y auditabilidad del proceso.
- Facilita simulacion de escenarios de decision en tiempo real.

## 8. Limitaciones

- Desempeno dependiente de la calidad y representatividad del dataset.
- Riesgo de sobreajuste si se aumenta complejidad sin control.
- Requiere mantenimiento de datos para conservar validez temporal.

## 9. Conclusiones

El proyecto demuestra que un arbol de decision puede apoyar eficazmente la seleccion de proveedores cuando se combina con una interfaz accesible y un pipeline robusto de preprocesamiento. La principal ventaja de la solucion es su interpretabilidad, ya que permite justificar las decisiones con reglas claras y variables de mayor impacto. Como linea futura, se recomienda comparar el modelo con ensambles como Random Forest y evaluar estabilidad temporal con nuevos periodos de datos.

## 10. Recomendaciones de mejora futura

- Incorporar validacion cruzada y optimizacion de hiperparametros.
- Anadir persistencia del modelo entrenado para despliegue estable.
- Integrar indicadores de costo total y riesgo contractual.
- Implementar monitoreo de deriva de datos y desempeno del modelo.

## 11. Referencias tecnicas sugeridas

- Breiman, L. et al. (1984). Classification and Regression Trees.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
- Documentacion oficial de Streamlit y scikit-learn.
