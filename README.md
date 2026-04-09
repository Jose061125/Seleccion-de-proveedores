# Sistema de Seleccion de Proveedores con Arboles de Decision

Proyecto de "Desarrollo e implementacion de un sistema de seleccion de proveedores basado en arboles de decision con Python y Streamlit".

## Objetivo

Construir una aplicacion que permita:

- Cargar datos de proveedores (CSV).
- Entrenar un modelo de clasificacion con arboles de decision.
- Evaluar el rendimiento del modelo con metricas clave.
- Interpretar decisiones mediante reglas del arbol e importancia de variables.
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
2. Elegir columna objetivo (por defecto `Seleccionado`).
3. Configurar variables predictoras y parametros del arbol.
4. Entrenar el modelo.
5. Revisar metricas, matriz de confusion, reporte e importancia de variables.
6. Consultar reglas del arbol para interpretabilidad.
7. Probar predicciones para un nuevo proveedor.

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
- El modelo es interpretable y orientado a toma de decision empresarial.

## Documentacion para Exposicion

- Explicacion funcional y tecnica: `docs/EXPLICACION_FUNCIONAMIENTO.md`
- Informe academico completo: `docs/INFORME_ACADEMICO.md`
- Ejemplo guiado de interfaz: `docs/EJEMPLO_GUIADO_INTERFAZ.md`