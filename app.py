from __future__ import annotations

import pandas as pd
import streamlit as st

from src.pipeline import decision_rules, evaluate_model, feature_importances, split_features_target, train_decision_tree


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
            <p>Motor de apoyo a decisiones basado en arboles de decision para compras, abastecimiento y gestion de riesgo.</p>
            <div class="chip-row">
                <span class="chip">Clasificacion supervisada</span>
                <span class="chip">Interpretabilidad</span>
                <span class="chip">Prediccion en tiempo real</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="status-ribbon">
        Panel interactivo para entrenar, evaluar e interpretar decisiones de seleccion de proveedores en una sola vista.
    </div>
    """,
    unsafe_allow_html=True,
)

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

profile_1, profile_2, profile_3 = st.columns(3)
with profile_1:
    st.markdown(
        f"""
        <div class="dataset-mini">
            <div class="label">Registros cargados</div>
            <div class="value">{len(df)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with profile_2:
    st.markdown(
        f"""
        <div class="dataset-mini">
            <div class="label">Columnas detectadas</div>
            <div class="value">{len(df.columns)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with profile_3:
    missing_values = int(df.isna().sum().sum())
    st.markdown(
        f"""
        <div class="dataset-mini">
            <div class="label">Valores faltantes</div>
            <div class="value">{missing_values}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Vista previa de datos")
st.dataframe(df.head(20), use_container_width=True)
st.caption(f"Registros cargados: {len(df)} | Columnas detectadas: {len(df.columns)} | Faltantes: {missing_values}")
st.markdown("</div>", unsafe_allow_html=True)

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

test_size = st.sidebar.slider("Tamano de prueba", min_value=0.1, max_value=0.4, step=0.05, value=0.2)
criterion = st.sidebar.selectbox("Criterio", options=["gini", "entropy", "log_loss"], index=0)
max_depth = st.sidebar.slider("Profundidad maxima", min_value=1, max_value=15, value=4)
min_samples_split = st.sidebar.slider("Minimo de muestras para dividir", min_value=2, max_value=20, value=2)

train_button = st.sidebar.button("Entrenar modelo")

if "trained" not in st.session_state:
    st.session_state.trained = False

if train_button:
    try:
        X, y = split_features_target(df, target_col, selected_features)
        artifacts = train_decision_tree(
            X,
            y,
            test_size=test_size,
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
        )

        result = evaluate_model(artifacts["pipeline"], artifacts["X_test"], artifacts["y_test"])

        st.session_state.pipeline = artifacts["pipeline"]
        st.session_state.selected_features = selected_features
        st.session_state.metrics = result["metrics"]
        st.session_state.report_df = result["report_df"]
        st.session_state.confusion_matrix = result["confusion_matrix"]
        st.session_state.importance_df = feature_importances(artifacts["pipeline"])
        st.session_state.rules = decision_rules(artifacts["pipeline"])
        st.session_state.feature_defaults = df[selected_features].copy()
        st.session_state.trained = True

    except ValueError as exc:
        st.error(f"Error durante el entrenamiento: {exc}")
        st.stop()

if st.session_state.trained:
    st.markdown('<h3 class="section-title">Resultados del modelo</h3>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Accuracy", f"{st.session_state.metrics['accuracy']:.3f}", "Desempeno global")
    with k2:
        render_kpi_card("Precision", f"{st.session_state.metrics['precision_weighted']:.3f}", "Promedio ponderado")
    with k3:
        render_kpi_card("Recall", f"{st.session_state.metrics['recall_weighted']:.3f}", "Cobertura ponderada")
    with k4:
        render_kpi_card("F1", f"{st.session_state.metrics['f1_weighted']:.3f}", "Balance precision/recall")

    tab_metrics, tab_interpret, tab_predict = st.tabs(["Rendimiento", "Interpretabilidad", "Prediccion"])

    with tab_metrics:
        m_left, m_right = st.columns([1, 1])
        with m_left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Matriz de confusion")
            st.dataframe(st.session_state.confusion_matrix, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with m_right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Reporte de clasificacion")
            st.dataframe(st.session_state.report_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_interpret:
        i_left, i_right = st.columns([1.1, 0.9])
        with i_left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Importancia de variables")
            top_importance = st.session_state.importance_df.head(15).set_index("feature")
            st.bar_chart(top_importance["importance"])
            st.dataframe(st.session_state.importance_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with i_right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Reglas del arbol")
            st.caption("Explicacion textual de las decisiones internas del modelo")
            st.code(st.session_state.rules)
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
                    default_value = values[0] if values else ""
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
else:
    st.info("Configura los parametros y pulsa 'Entrenar modelo' para comenzar.")
