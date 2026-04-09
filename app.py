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
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Bitter:wght@500;700&display=swap');

        :root {
            --ink: #11242b;
            --text: #153640;
            --accent: #0f9d8f;
            --warm: #f0a03b;
            --mist: #f4f8f9;
            --panel: rgba(255, 255, 255, 0.78);
            --line: #bfd4d8;
        }

        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(1200px 420px at 5% -20%, rgba(15,157,143,0.18), transparent 65%),
                radial-gradient(920px 360px at 90% -15%, rgba(240,160,59,0.20), transparent 70%),
                linear-gradient(180deg, #f6fafb 0%, #eef5f5 60%, #f8faf8 100%);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d3542 0%, #164757 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.14);
        }

        section[data-testid="stSidebar"] * {
            color: #e8f7f6;
        }

        .hero-card {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.15rem 1.4rem;
            background: linear-gradient(130deg, rgba(255,255,255,0.94) 0%, rgba(232,245,245,0.92) 100%);
            box-shadow: 0 12px 28px rgba(10, 33, 43, 0.08);
            margin-bottom: 1rem;
            animation: rise .55s ease-out;
        }

        .status-ribbon {
            border: 1px solid rgba(15,157,143,0.35);
            border-radius: 12px;
            padding: .55rem .8rem;
            margin: 0 0 1rem;
            background: linear-gradient(90deg, rgba(15,157,143,0.10), rgba(240,160,59,0.10));
            font-size: .88rem;
            color: #184650;
        }

        .dataset-mini {
            border: 1px solid #c9dde0;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.85);
            padding: .65rem .8rem;
            margin-bottom: .7rem;
        }

        .dataset-mini .label {
            font-size: .78rem;
            color: #4a6a74;
            margin-bottom: .12rem;
        }

        .dataset-mini .value {
            font-size: 1.12rem;
            font-weight: 700;
            color: #0d5d66;
        }

        .kpi-tile {
            border: 1px solid #c3dadc;
            border-radius: 14px;
            padding: .62rem .85rem;
            background: linear-gradient(160deg, rgba(255,255,255,0.95), rgba(233,245,244,0.95));
            box-shadow: 0 8px 18px rgba(8, 41, 45, 0.07);
            height: 100%;
            animation: rise .6s ease-out;
        }

        .kpi-title {
            font-size: .78rem;
            color: #3f6670;
            margin-bottom: .1rem;
        }

        .kpi-value {
            font-size: 1.6rem;
            line-height: 1.15;
            color: #0c6c73;
            font-weight: 700;
        }

        .kpi-subtitle {
            margin-top: .2rem;
            font-size: .74rem;
            color: #53757e;
        }

        .section-title {
            font-family: 'Bitter', serif;
            color: #16343d;
            letter-spacing: 0.2px;
            margin-bottom: .35rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: .35rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.75);
            border: 1px solid #c9dcdf;
            border-radius: 10px;
            padding: .4rem .8rem;
        }

        .stTabs [aria-selected="true"] {
            border-color: #67aba8;
            background: rgba(15,157,143,0.14);
        }

        .hero-card h1 {
            font-family: 'Bitter', serif;
            letter-spacing: 0.2px;
            margin: 0;
            color: var(--ink);
            font-size: clamp(1.5rem, 2.6vw, 2.2rem);
        }

        .hero-card p {
            margin: .35rem 0 0;
            font-size: .97rem;
            color: #335864;
        }

        .chip-row {
            display: flex;
            gap: .45rem;
            flex-wrap: wrap;
            margin-top: .65rem;
        }

        .chip {
            border: 1px solid rgba(15,157,143,0.35);
            border-radius: 999px;
            padding: .18rem .58rem;
            font-size: .76rem;
            color: #0b6f67;
            background: rgba(15,157,143,0.08);
        }

        .panel {
            border: 1px solid #d0e1e3;
            border-radius: 14px;
            background: var(--panel);
            box-shadow: 0 10px 22px rgba(9, 33, 41, 0.05);
            padding: .55rem .8rem .8rem;
            margin-bottom: .9rem;
            backdrop-filter: blur(4px);
        }

        [data-testid="stMetricValue"] {
            color: #0b6f67;
            font-weight: 700;
        }

        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #c8dcdf;
        }

        @keyframes rise {
            from { transform: translateY(8px); opacity: 0; }
            to   { transform: translateY(0); opacity: 1; }
        }

        @media (max-width: 900px) {
            .hero-card {
                padding: 1rem 1rem;
            }
            .panel {
                padding: .5rem .55rem .65rem;
            }
            .kpi-value {
                font-size: 1.35rem;
            }
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
