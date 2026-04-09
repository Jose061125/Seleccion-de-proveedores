from __future__ import annotations

import io
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.pipeline import (
    MODEL_LABELS,
    compare_models,
    cross_validation_summary,
    decision_rules,
    evaluate_model,
    feature_importances,
    split_features_target,
    train_model,
    tree_visualization_figure,
)


MODEL_FILE_PATH = Path("models/modelo_proveedores.joblib")


def render_kpi_card(title: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-tile">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_card(label: str, value: str | int) -> None:
    st.markdown(
        f"""
        <div class="dataset-mini">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def serialize_model_bundle(bundle: dict) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer)
    return buffer.getvalue()


def update_session_from_bundle(bundle: dict, df: pd.DataFrame) -> None:
    required_features = bundle["selected_features"]
    missing_features = [feature for feature in required_features if feature not in df.columns]
    if missing_features:
        raise ValueError(
            "El dataset actual no contiene las columnas requeridas por el modelo: "
            + ", ".join(missing_features)
        )

    st.session_state.pipeline = bundle["pipeline"]
    st.session_state.selected_features = bundle["selected_features"]
    st.session_state.target_col = bundle.get("target_col", "Seleccionado")
    st.session_state.model_name = bundle.get("model_name", "decision_tree")
    st.session_state.model_label = MODEL_LABELS.get(st.session_state.model_name, st.session_state.model_name)
    st.session_state.metrics = bundle.get("metrics", {})
    st.session_state.report_df = bundle.get("report_df", pd.DataFrame())
    st.session_state.confusion_matrix = bundle.get("confusion_matrix", pd.DataFrame())
    st.session_state.importance_df = bundle.get("importance_df", pd.DataFrame())
    st.session_state.rules = bundle.get("rules", "")
    st.session_state.cv_summary_df = bundle.get("cv_summary_df", pd.DataFrame())
    st.session_state.comparison_df = bundle.get("comparison_df", pd.DataFrame())
    st.session_state.feature_defaults = df[required_features].copy()
    st.session_state.trained = True


st.set_page_config(page_title="Seleccion de Proveedores", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Poppins:wght@700;800;900&display=swap');

    :root {
        --bg:      #f0f4ff;
        --surface: #ffffff;
        --border:  #d4daf5;
        --ink:     #070b22;
        --text:    #17233d;
        --muted:   #4b5c7c;
        --accent:  #4f46e5;
        --accent2: #7c3aed;
        --warm:    #f59e0b;
        --ok:      #10b981;
        --panel:   #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 15.5px;
        line-height: 1.7;
        color: var(--text);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Poppins', sans-serif;
        color: var(--ink);
        letter-spacing: -0.5px;
        line-height: 1.25;
        text-shadow: 0 1px 0 rgba(255, 255, 255, 0.35);
    }

    h2 { font-size: 1.45rem; font-weight: 800; }
    h3 { font-size: 1.2rem;  font-weight: 700; }
    h4 { font-size: 1.05rem; font-weight: 700; letter-spacing: -0.2px; }

    [data-testid="stHeadingWithActionElements"] h1,
    [data-testid="stHeadingWithActionElements"] h2,
    [data-testid="stHeadingWithActionElements"] h3,
    [data-testid="stHeadingWithActionElements"] h4,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: var(--ink) !important;
        opacity: 1 !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 800 !important;
    }

    p, li, span, td, th, label {
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }

    [data-testid="stAppViewContainer"] {
        background: var(--bg);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1535 0%, #1a2550 100%);
        border-right: 1px solid rgba(79,70,229,0.25);
    }

    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #e2e8f8 !important;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-family: 'Poppins', sans-serif;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 14px;
        border: none;
        border-radius: 10px;
        padding: .5rem 1.2rem;
        transition: opacity .2s;
    }

    .stButton > button:hover {
        opacity: .88;
    }

    .hero-card {
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.4rem 1.8rem;
        background: linear-gradient(135deg, #ffffff 0%, #eef0fe 100%);
        box-shadow: 0 4px 24px rgba(79,70,229,0.10);
        margin-bottom: 1.2rem;
        animation: rise .5s ease-out;
    }

    .hero-card h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 900;
        font-size: clamp(1.55rem, 2.8vw, 2.25rem);
        color: var(--ink);
        letter-spacing: -0.8px;
        line-height: 1.15;
        margin: 0 0 .45rem;
    }

    .hero-card p {
        font-size: 1.02rem;
        font-weight: 400;
        color: var(--muted);
        line-height: 1.65;
        margin: 0;
    }

    .chip-row {
        display: flex;
        gap: .4rem;
        flex-wrap: wrap;
        margin-top: .75rem;
    }

    .chip {
        border: 1px solid rgba(79,70,229,0.30);
        border-radius: 999px;
        padding: .2rem .65rem;
        font-size: .75rem;
        font-weight: 500;
        color: var(--accent);
        background: rgba(79,70,229,0.07);
    }

    .status-ribbon {
        border-left: 4px solid var(--accent);
        border-radius: 0 10px 10px 0;
        padding: .6rem 1.1rem;
        margin: 0 0 1.2rem;
        background: rgba(79,70,229,0.06);
        font-size: .92rem;
        font-weight: 500;
        font-style: italic;
        color: var(--ink);
        letter-spacing: .1px;
    }

    .dataset-mini {
        border: 1px solid var(--border);
        border-radius: 14px;
        background: var(--surface);
        padding: .75rem 1rem;
        margin-bottom: .8rem;
        box-shadow: 0 2px 8px rgba(15,21,53,0.05);
    }

    .dataset-mini .label {
        font-family: 'Inter', sans-serif;
        font-size: .72rem;
        font-weight: 700;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: .9px;
        margin-bottom: .2rem;
    }

    .dataset-mini .value {
        font-family: 'Poppins', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: var(--ink);
    }

    .kpi-tile {
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: .85rem 1rem;
        background: var(--surface);
        box-shadow: 0 4px 16px rgba(15,21,53,0.07);
        height: 100%;
        animation: rise .55s ease-out;
    }

    .kpi-title {
        font-family: 'Inter', sans-serif;
        font-size: .72rem;
        font-weight: 700;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: .25rem;
    }

    .kpi-value {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        line-height: 1.05;
        color: var(--accent);
        font-weight: 800;
        letter-spacing: -1px;
    }

    .kpi-subtitle {
        margin-top: .3rem;
        font-size: .75rem;
        font-weight: 500;
        color: var(--muted);
    }

    .section-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        color: var(--ink);
        font-size: 1.25rem;
        letter-spacing: -0.4px;
        margin-bottom: .5rem;
    }

    .panel {
        border: 1px solid var(--border);
        border-radius: 16px;
        background: var(--panel);
        box-shadow: 0 4px 16px rgba(15,21,53,0.05);
        padding: 1rem 1.1rem 1.1rem;
        margin-bottom: 1rem;
    }

    .panel h4, .panel h3 {
        font-family: 'Poppins', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -0.2px;
        margin-bottom: .55rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: .4rem;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: .5rem 1rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: .88rem;
        color: var(--muted);
        letter-spacing: .1px;
    }

    .stTabs [aria-selected="true"] {
        border-color: var(--accent);
        background: rgba(79,70,229,0.09);
        color: var(--accent) !important;
        font-weight: 700;
    }

    [data-testid="stMetricValue"] {
        color: var(--accent);
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }

    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--border);
        font-family: 'Inter', sans-serif;
        font-size: 13.5px;
        line-height: 1.6;
    }

    [data-testid="stDataFrame"] th {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 12.5px;
        text-transform: uppercase;
        letter-spacing: .5px;
    }

    code, pre {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 13px;
        line-height: 1.7;
    }

    @keyframes rise {
        from { transform: translateY(6px); opacity: 0; }
        to   { transform: translateY(0);   opacity: 1; }
    }

    @media (max-width: 900px) {
        .hero-card { padding: 1rem 1.1rem; }
        .kpi-value { font-size: 1.5rem; }
        .panel     { padding: .7rem .8rem .9rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1>Sistema Inteligente de Seleccion de Proveedores</h1>
        <p>Motor de apoyo a decisiones basado en modelos de clasificacion para compras, abastecimiento y gestion de riesgo.</p>
        <div class="chip-row">
            <span class="chip">Comparacion de modelos</span>
            <span class="chip">Interpretabilidad</span>
            <span class="chip">Exportacion y persistencia</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="status-ribbon">
        Panel interactivo para entrenar, comparar, validar y reutilizar modelos de seleccion de proveedores desde una sola interfaz.
    </div>
    """,
    unsafe_allow_html=True,
)

if "trained" not in st.session_state:
    st.session_state.trained = False

st.sidebar.header("Configuracion")
data_source = st.sidebar.radio("Fuente de datos", ["Ejemplo incluido", "Subir CSV"], horizontal=False)

if data_source == "Subir CSV":
    uploaded_file = st.sidebar.file_uploader("Selecciona archivo CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Sube un archivo CSV para continuar o cambia a 'Ejemplo incluido'.")
        st.stop()
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/proveedores_ejemplo.csv")

if df.empty:
    st.error("El conjunto de datos esta vacio.")
    st.stop()

all_columns = df.columns.tolist()
default_target_index = all_columns.index("Seleccionado") if "Seleccionado" in all_columns else len(all_columns) - 1
target_col = st.sidebar.selectbox("Columna objetivo (target)", options=all_columns, index=default_target_index)

candidate_features = [col for col in all_columns if col != target_col]
selected_features = st.sidebar.multiselect(
    "Columnas predictoras",
    options=candidate_features,
    default=candidate_features,
)

if len(selected_features) == 0:
    st.warning("Selecciona al menos una columna predictora.")
    st.stop()

model_name = st.sidebar.selectbox(
    "Modelo principal",
    options=list(MODEL_LABELS.keys()),
    format_func=lambda item: MODEL_LABELS[item],
)

test_size = st.sidebar.slider("Tamano de prueba", min_value=0.1, max_value=0.4, step=0.05, value=0.2)
max_depth = st.sidebar.slider("Profundidad maxima", min_value=1, max_value=15, value=4)
criterion = st.sidebar.selectbox("Criterio del arbol", options=["gini", "entropy", "log_loss"], index=0)
min_samples_split = st.sidebar.slider("Minimo de muestras para dividir", min_value=2, max_value=20, value=2)
max_cv_folds = max(2, min(8, int(df[target_col].value_counts().min())))
cv_folds = st.sidebar.slider("Folds de validacion cruzada", min_value=2, max_value=max_cv_folds, value=min(5, max_cv_folds))
st.sidebar.caption("La profundidad y el criterio aplican principalmente a modelos basados en arboles.")

uploaded_model = st.sidebar.file_uploader("Cargar modelo guardado (.joblib)", type=["joblib"])
load_model_button = st.sidebar.button("Cargar modelo guardado", use_container_width=True)
train_button = st.sidebar.button("Entrenar modelo", use_container_width=True)

if st.session_state.trained and "selected_features" in st.session_state:
    missing_current_features = [feature for feature in st.session_state.selected_features if feature not in df.columns]
    if missing_current_features:
        st.session_state.trained = False

if load_model_button:
    if uploaded_model is None:
        st.sidebar.warning("Selecciona un archivo .joblib antes de cargarlo.")
    else:
        try:
            bundle = joblib.load(io.BytesIO(uploaded_model.getvalue()))
            update_session_from_bundle(bundle, df)
            st.sidebar.success("Modelo cargado correctamente.")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"No se pudo cargar el modelo: {exc}")

if train_button:
    try:
        X, y = split_features_target(df, target_col, selected_features)
        artifacts = train_model(
            X,
            y,
            model_name=model_name,
            test_size=test_size,
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
        )
        result = evaluate_model(artifacts["pipeline"], artifacts["X_test"], artifacts["y_test"])
        cv_summary_df = cross_validation_summary(
            X,
            y,
            model_name=model_name,
            cv_folds=cv_folds,
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
        )
        comparison_df = compare_models(
            X,
            y,
            cv_folds=cv_folds,
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
        )
        importance_df = feature_importances(artifacts["pipeline"])
        rules = decision_rules(artifacts["pipeline"])

        bundle = {
            "pipeline": artifacts["pipeline"],
            "model_name": model_name,
            "target_col": target_col,
            "selected_features": selected_features,
            "metrics": result["metrics"],
            "report_df": result["report_df"],
            "confusion_matrix": result["confusion_matrix"],
            "importance_df": importance_df,
            "rules": rules,
            "cv_summary_df": cv_summary_df,
            "comparison_df": comparison_df,
        }
        update_session_from_bundle(bundle, df)

    except ValueError as exc:
        st.error(f"Error durante el entrenamiento: {exc}")
        st.stop()

missing_values = int(df.isna().sum().sum())
profile_1, profile_2, profile_3, profile_4 = st.columns(4)
with profile_1:
    render_dataset_card("Registros cargados", len(df))
with profile_2:
    render_dataset_card("Columnas detectadas", len(df.columns))
with profile_3:
    render_dataset_card("Valores faltantes", missing_values)
with profile_4:
    render_dataset_card("Modelo activo", MODEL_LABELS.get(model_name, model_name))

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Vista previa de datos")
st.dataframe(df.head(20), use_container_width=True)
st.caption(
    f"Registros cargados: {len(df)} | Columnas detectadas: {len(df.columns)} | Faltantes: {missing_values} | "
    f"Target actual: {target_col}"
)
st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.trained:
    current_model_label = st.session_state.get("model_label", MODEL_LABELS.get(model_name, model_name))
    st.markdown('<h3 class="section-title">Resultados del modelo</h3>', unsafe_allow_html=True)
    st.caption(f"Modelo entrenado o cargado: {current_model_label}")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Accuracy", f"{st.session_state.metrics['accuracy']:.3f}", "Desempeno global")
    with k2:
        render_kpi_card("Precision", f"{st.session_state.metrics['precision_weighted']:.3f}", "Promedio ponderado")
    with k3:
        render_kpi_card("Recall", f"{st.session_state.metrics['recall_weighted']:.3f}", "Cobertura ponderada")
    with k4:
        render_kpi_card("F1", f"{st.session_state.metrics['f1_weighted']:.3f}", "Balance precision/recall")

    tab_metrics, tab_interpret, tab_predict, tab_manage = st.tabs(
        ["Rendimiento", "Interpretabilidad", "Prediccion", "Gestion"]
    )

    with tab_metrics:
        top_left, top_right = st.columns([1, 1])
        with top_left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Matriz de confusion")
            st.dataframe(st.session_state.confusion_matrix, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with top_right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Reporte de clasificacion")
            st.dataframe(st.session_state.report_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        bottom_left, bottom_right = st.columns([1, 1])
        with bottom_left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Validacion cruzada")
            st.dataframe(st.session_state.cv_summary_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with bottom_right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Comparacion de modelos")
            st.dataframe(st.session_state.comparison_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_interpret:
        i_left, i_right = st.columns([1.1, 0.9])
        with i_left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Importancia de variables")
            if st.session_state.importance_df.empty:
                st.info("El modelo actual no expone importancia de variables interpretable.")
            else:
                top_importance = st.session_state.importance_df.head(15).set_index("feature")
                st.bar_chart(top_importance["importance"])
                st.dataframe(st.session_state.importance_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with i_right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Reglas del modelo")
            st.caption("Disponibles en detalle solo para arbol de decision.")
            st.code(st.session_state.rules)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Visualizacion del arbol")
        tree_fig = tree_visualization_figure(st.session_state.pipeline, max_depth=min(3, max_depth))
        if tree_fig is None:
            st.info("La vista grafica del arbol solo esta disponible para el modelo Arbol de decision.")
        else:
            st.pyplot(tree_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_predict:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Prediccion de un proveedor")
        with st.form("prediction_form"):
            input_data: dict[str, object] = {}

            for feature in st.session_state.selected_features:
                series = st.session_state.feature_defaults[feature]
                if pd.api.types.is_numeric_dtype(series):
                    default_value = float(series.median())
                    input_data[feature] = st.number_input(feature, value=default_value)
                else:
                    values = sorted(series.dropna().astype(str).unique().tolist())
                    input_data[feature] = st.selectbox(feature, options=values, index=0 if values else None)

            submit_prediction = st.form_submit_button("Predecir")

        if submit_prediction:
            new_df = pd.DataFrame([input_data])
            prediction = st.session_state.pipeline.predict(new_df)[0]
            st.success(f"Prediccion del modelo: {prediction}")

            if hasattr(st.session_state.pipeline, "predict_proba"):
                probs = st.session_state.pipeline.predict_proba(new_df)[0]
                classes = st.session_state.pipeline.classes_
                probs_df = pd.DataFrame({"Clase": classes, "Probabilidad": probs}).sort_values(
                    "Probabilidad", ascending=False
                )
                st.dataframe(probs_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_manage:
        model_bundle = {
            "pipeline": st.session_state.pipeline,
            "model_name": st.session_state.model_name,
            "target_col": st.session_state.target_col,
            "selected_features": st.session_state.selected_features,
            "metrics": st.session_state.metrics,
            "report_df": st.session_state.report_df,
            "confusion_matrix": st.session_state.confusion_matrix,
            "importance_df": st.session_state.importance_df,
            "rules": st.session_state.rules,
            "cv_summary_df": st.session_state.cv_summary_df,
            "comparison_df": st.session_state.comparison_df,
        }
        model_bytes = serialize_model_bundle(model_bundle)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Guardar o reutilizar modelo")
        st.download_button(
            label="Descargar modelo entrenado",
            data=model_bytes,
            file_name="modelo_proveedores.joblib",
            mime="application/octet-stream",
            use_container_width=True,
        )
        if st.button("Guardar copia local en disco", use_container_width=True):
            MODEL_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            MODEL_FILE_PATH.write_bytes(model_bytes)
            st.success(f"Modelo guardado en {MODEL_FILE_PATH}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Exportar resultados")
        export_col_1, export_col_2 = st.columns(2)
        with export_col_1:
            st.download_button(
                label="Descargar reporte de clasificacion",
                data=dataframe_to_csv_bytes(st.session_state.report_df),
                file_name="reporte_clasificacion.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                label="Descargar validacion cruzada",
                data=dataframe_to_csv_bytes(st.session_state.cv_summary_df),
                file_name="validacion_cruzada.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with export_col_2:
            st.download_button(
                label="Descargar matriz de confusion",
                data=dataframe_to_csv_bytes(st.session_state.confusion_matrix),
                file_name="matriz_confusion.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                label="Descargar comparacion de modelos",
                data=dataframe_to_csv_bytes(st.session_state.comparison_df),
                file_name="comparacion_modelos.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Configura los parametros y pulsa 'Entrenar modelo' para comenzar.")
