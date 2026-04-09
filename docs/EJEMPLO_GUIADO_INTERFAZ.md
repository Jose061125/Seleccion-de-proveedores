# Ejemplo Guiado para Entender la Interfaz

Este documento presenta un ejemplo practico, paso a paso, para comprender como usar la interfaz del sistema de seleccion de proveedores.

## Objetivo del ejemplo

Simular una sesion real donde un analista de compras:

- carga datos,
- entrena el modelo,
- interpreta resultados,
- y predice si un nuevo proveedor debe ser seleccionado.

## Escenario

Supongamos que queremos evaluar un proveedor nuevo para decidir si conviene aprobarlo en el proceso de compras.

## Paso 1. Abrir la app

1. Ejecutar:

```bash
streamlit run app.py
```

2. En la barra lateral, elegir:

- Fuente de datos: Ejemplo incluido.

Que ocurre en la interfaz:

- Se carga automaticamente el archivo de ejemplo con historico de proveedores.

## Paso 2. Revisar la tabla de datos

En la zona principal aparece la vista previa del dataset.

Como explicarlo:

- Cada fila representa un proveedor historico.
- Cada columna representa una caracteristica (precio, calidad, etc.).
- La columna objetivo indica si en el pasado ese proveedor fue seleccionado.

### Significado de cada columna del CSV

- Proveedor:
	nombre o identificador del proveedor.
- Precio:
	costo ofertado por el proveedor.
- Calidad:
	calificacion numerica de calidad del proveedor o de su oferta.
- TiempoEntregaDias:
	cantidad de dias que tarda en entregar.
- CumplimientoHistorico:
	indicador historico de cumplimiento, normalmente expresado como porcentaje.
- CertificacionISO:
	indica si el proveedor cuenta con certificacion ISO.
- Ubicacion:
	procedencia geografica del proveedor, por ejemplo Local, Nacional, Regional o Internacional.
- Seleccionado:
	indica si el proveedor fue aprobado o no en decisiones anteriores.

### Variable objetivo

La variable objetivo del modelo es `Seleccionado`.

Esto significa que el sistema usa el resto de columnas como variables de entrada para predecir si un proveedor debe clasificarse como seleccionado (`Si`) o no seleccionado (`No`).

## Paso 3. Configurar el problema de clasificacion

En la barra lateral:

- Columna objetivo: Seleccionado.
- Columnas predictoras: Precio, Calidad, TiempoEntregaDias, CumplimientoHistorico, CertificacionISO, Ubicacion.
- Criterio del arbol: gini.
- Profundidad maxima: 4.
- Minimo de muestras para dividir: 2.
- Tamano de prueba: 0.20.

Que significa:

- El modelo aprende con 80% de datos y se evalua con 20%.
- La profundidad limita complejidad para evitar sobreajuste.

## Paso 4. Entrenar el modelo

1. Presionar el boton Entrenar modelo.

Que hace el sistema internamente:

- Separa variables de entrada y variable objetivo.
- Imputa faltantes numericos con mediana.
- Imputa categoricos con valor mas frecuente.
- Convierte categoricos a formato numerico con one-hot encoding.
- Entrena el arbol de decision.

## Paso 5. Interpretar resultados en pantalla

La interfaz muestra varios bloques:

1. Tarjetas de metricas:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 (weighted)

2. Matriz de confusion:
- Muestra aciertos y errores por clase.

3. Reporte de clasificacion:
- Detalla precision, recall y F1 por clase.

4. Importancia de variables:
- Indica que variables influyen mas en la decision.

5. Reglas del arbol:
- Explica la logica de decision en formato si-entonces.

## Paso 6. Hacer una prediccion de un proveedor nuevo

En el formulario final, ingresar por ejemplo:

- Precio: 10000
- Calidad: 8.5
- TiempoEntregaDias: 8
- CumplimientoHistorico: 94
- CertificacionISO: Si
- Ubicacion: Nacional

Presionar Predecir.

Salida esperada:

- Prediccion del modelo: Si o No.
- Probabilidad por clase (cuando esta disponible).

## Como explicarlo en una exposicion

Mensaje corto:

- El usuario configura y entrena el modelo desde la barra lateral.
- La app muestra metricas de desempeno para validar confiabilidad.
- El arbol es interpretable porque muestra variables importantes y reglas.
- Finalmente, se pueden simular nuevos proveedores en tiempo real.

## Guion de 60 segundos

"Esta interfaz permite convertir datos historicos de proveedores en una decision objetiva de seleccion. Primero cargamos los datos, definimos la variable objetivo y configuramos el arbol de decision. Luego entrenamos el modelo y revisamos metricas como accuracy, precision, recall y F1, junto con la matriz de confusion. Lo mas importante es que el sistema no solo predice, tambien explica por que predice, gracias a la importancia de variables y las reglas del arbol. Finalmente, ingresamos un proveedor nuevo y obtenemos una recomendacion inmediata para apoyar decisiones de compras con mayor trazabilidad y menor subjetividad."