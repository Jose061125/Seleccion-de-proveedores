# Sistema de Seleccion de Proveedores con Arboles de Decision

Proyecto de "Desarrollo e implementacion de un sistema de seleccion de proveedores basado en arboles de decision con Python y Streamlit".

## Objetivo

Construir una aplicacion que permita:

- Cargar datos de proveedores (CSV).
- Entrenar un modelo de clasificacion principal.
- Comparar varios modelos de machine learning.
- Evaluar el rendimiento con metricas y validacion cruzada.
- Interpretar decisiones mediante reglas del arbol, grafico e importancia de variables.
- Guardar, cargar y exportar resultados del modelo.
- Realizar predicciones sobre nuevos proveedores.

## Stack Tecnologico

- Python 3.10+
- Streamlit
- Pandas
- Scikit-learn

## Estructura del Proyecto

.
|-- app.py
|-- data/
|   `-- proveedores_ejemplo.csv
|-- src/
|   |-- __init__.py
|   `-- pipeline.py
|-- requirements.txt
`-- README.md

## Instalacion

1. Crear y activar un entorno virtual (opcional pero recomendado):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecucion

```bash
streamlit run app.py
```

La aplicacion abrira en el navegador y te permitira usar:

- Dataset de ejemplo incluido en `data/proveedores_ejemplo.csv`.
- O un CSV propio cargado desde la barra lateral.

## Flujo de Uso en la App

1. Seleccionar fuente de datos (ejemplo o archivo CSV).
2. Elegir columna objetivo y variables predictoras.
3. Seleccionar modelo principal y parametros de entrenamiento.
4. Ejecutar entrenamiento y validacion cruzada.
5. Revisar metricas, matriz de confusion y reporte de clasificacion.
6. Comparar el rendimiento entre Arbol de decision, Random Forest y Regresion logistica.
7. Consultar importancia de variables, reglas y visualizacion del arbol.
8. Probar predicciones para un nuevo proveedor.
9. Descargar resultados o guardar/cargar el modelo entrenado.

## Formato Recomendado del CSV

Debe incluir una columna objetivo de clasificacion (por ejemplo, `Seleccionado`) y columnas predictoras numericas y/o categoricas.

Variable objetivo por defecto:

- `Seleccionado`: indica si historicamente el proveedor fue aprobado (`Si`) o no (`No`).

Ejemplo de columnas:

- `Proveedor`
- `Precio`
- `Calidad`
- `TiempoEntregaDias`
- `CumplimientoHistorico`
- `CertificacionISO`
- `Ubicacion`
- `Seleccionado`

## Notas

- El preprocesamiento se realiza con `ColumnTransformer`:
	- Imputacion de valores faltantes numericos con mediana.
	- Imputacion de categoricos con valor mas frecuente.
	- Codificacion one-hot para columnas categoricas.
- La aplicacion incluye comparacion automatica entre tres modelos:
	- Arbol de decision.
	- Random Forest.
	- Regresion logistica.
- Se incorpora validacion cruzada para evaluar estabilidad del modelo.
- Es posible descargar el modelo entrenado en formato `.joblib` y volver a cargarlo.
- Se pueden exportar reportes, matriz de confusion, comparacion de modelos y validacion cruzada en CSV.
- El modelo sigue orientado a toma de decision empresarial e interpretabilidad.

## Documentacion para Exposicion

- Explicacion funcional y tecnica: `docs/EXPLICACION_FUNCIONAMIENTO.md`
- Informe academico completo: `docs/INFORME_ACADEMICO.md`
- Ejemplo guiado de interfaz: `docs/EJEMPLO_GUIADO_INTERFAZ.md`