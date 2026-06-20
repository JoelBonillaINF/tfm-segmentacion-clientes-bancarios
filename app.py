from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from segmentation_core import (
    InputValidationError,
    REQUIRED_TRANSACTION_COLUMNS,
    load_bundle,
    score_transactions,
)

st.set_page_config(
    page_title="Segmentación de clientes bancarios",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "artifacts" / "segmentation_bundle.joblib"
EXAMPLE_PATH = BASE_DIR / "data" / "cliente_ejemplo.csv"


@st.cache_resource
def get_bundle():
    return load_bundle(BUNDLE_PATH)


st.markdown(
    """
    <style>
    .main-title {font-size: 2.2rem; font-weight: 750; color: #0B2E59; margin-bottom: 0.15rem;}
    .subtitle {font-size: 1.05rem; color: #486581; margin-bottom: 1.2rem;}
    .result-card {padding: 1.1rem; border: 1px solid #D6E2EE; border-radius: 14px; background: #F8FBFE;}
    .segment-name {font-size: 1.35rem; font-weight: 750; color: #0B2E59;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">Asignación de nuevos clientes a segmentos</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Prototipo operativo del modelo K-Means con k = 4 desarrollado en el TFM.</div>',
    unsafe_allow_html=True,
)

try:
    bundle = get_bundle()
except Exception as exc:
    st.error(
        "No se pudo cargar el paquete del modelo. Ejecute primero el notebook de entrenamiento "
        "y confirme que exista artifacts/segmentation_bundle.joblib."
    )
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("Configuración")
    st.write(f"Modelo: **K-Means, k = {bundle['model'].n_clusters}**")
    st.write(f"Mínimo de transacciones: **{bundle['min_transactions']}**")
    st.write(f"Mínimo de días observados: **{bundle['min_lifetime_days']}**")

    use_training_reference = st.checkbox(
        "Usar fecha de corte del modelo",
        value=True,
        help=(
            "Recomendado para reproducir el análisis histórico. Para datos de otro periodo, "
            "debe evaluarse el reentrenamiento del modelo."
        ),
    )

    training_reference = pd.Timestamp(bundle["training_reference_date"])
    if use_training_reference:
        reference_date = training_reference
        st.caption(f"Fecha de corte: {training_reference}")
    else:
        selected_date = st.date_input(
            "Fecha de corte del análisis",
            value=pd.Timestamp.today().date(),
        )
        reference_date = pd.Timestamp(selected_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        st.warning(
            "La fecha de corte difiere de la usada al entrenar. Para uso productivo con datos actuales, "
            "conviene reentrenar el modelo sobre una ventana contemporánea."
        )

st.info(
    "La aplicación no fuerza una asignación cuando el cliente tiene menos de "
    f"{bundle['min_transactions']} transacciones o menos de "
    f"{bundle['min_lifetime_days']} días observados."
)

input_tab, manual_tab, format_tab = st.tabs(
    ["Importar CSV", "Introducción manual", "Formato requerido"]
)

transactions = None
source_label = None

with input_tab:
    uploaded = st.file_uploader(
        "Seleccione un CSV con transacciones de uno o varios clientes",
        type=["csv"],
    )
    if uploaded is not None:
        try:
            transactions = pd.read_csv(uploaded)
            source_label = uploaded.name
            st.success(f"Archivo cargado: {uploaded.name}")
            st.dataframe(transactions.head(20), use_container_width=True)
        except Exception as exc:
            st.error("No se pudo leer el archivo CSV.")
            st.exception(exc)

    if EXAMPLE_PATH.exists():
        example_bytes = EXAMPLE_PATH.read_bytes()
        st.download_button(
            "Descargar archivo de ejemplo",
            data=example_bytes,
            file_name="cliente_ejemplo.csv",
            mime="text/csv",
        )

with manual_tab:
    st.write(
        "Puede escribir o pegar transacciones desde Excel. Use el botón **+** de la tabla "
        "para añadir filas. Para ser elegible, un cliente necesita al menos 30 transacciones "
        "y 30 días observados."
    )
    manual_seed = pd.DataFrame(
        {
            "cc_num": ["CLIENTE_MANUAL"],
            "trans_date_trans_time": ["2020-01-01 10:00:00"],
            "amt": [50.0],
            "category": ["grocery_pos"],
            "merchant": ["Comercio_01"],
        }
    )
    manual_data = st.data_editor(
        manual_seed,
        num_rows="dynamic",
        use_container_width=True,
        key="manual_transactions",
    )
    if st.checkbox("Usar los datos introducidos manualmente"):
        transactions = manual_data
        source_label = "Entrada manual"

with format_tab:
    st.write("El archivo debe contener exactamente estas columnas mínimas:")
    st.code(",".join(REQUIRED_TRANSACTION_COLUMNS), language="text")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "cc_num": "CLIENTE_001",
                    "trans_date_trans_time": "2020-01-01 10:30:00",
                    "amt": 42.50,
                    "category": "grocery_pos",
                    "merchant": "Comercio_A",
                },
                {
                    "cc_num": "CLIENTE_001",
                    "trans_date_trans_time": "2020-01-05 18:45:00",
                    "amt": 85.20,
                    "category": "shopping_pos",
                    "merchant": "Comercio_B",
                },
            ]
        ),
        use_container_width=True,
    )

analyze = st.button(
    "Analizar y asignar segmentos",
    type="primary",
    use_container_width=True,
    disabled=transactions is None,
)

if analyze and transactions is not None:
    try:
        results, features, audit = score_transactions(
            transactions=transactions,
            reference_date=reference_date,
            bundle=bundle,
        )

        st.subheader("Validación del archivo")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Filas recibidas", audit["original_rows"])
        col2.metric("Duplicados eliminados", audit["duplicate_rows_removed"])
        col3.metric("Transacciones válidas", audit["clean_rows"])
        col4.metric("Clientes", audit["customers"])

        st.subheader("Resultado de segmentación")
        display_columns = [
            "cc_num",
            "status",
            "segment_id",
            "segment_name",
            "frequency",
            "customer_lifetime_days",
            "recency_days",
            "assignment_assessment",
        ]
        st.dataframe(results[display_columns], use_container_width=True, hide_index=True)

        for _, row in results.iterrows():
            title = f"Cliente {row['cc_num']}"
            with st.expander(title, expanded=len(results) == 1):
                if not bool(row["eligible"]):
                    st.warning(
                        "Historial insuficiente. No se asignó un cluster para evitar una clasificación poco robusta."
                    )
                    st.write(
                        f"Transacciones observadas: **{int(row['frequency'])}** / "
                        f"mínimo requerido: **{bundle['min_transactions']}**"
                    )
                    st.write(
                        f"Días observados: **{int(row['customer_lifetime_days'])}** / "
                        f"mínimo requerido: **{bundle['min_lifetime_days']}**"
                    )
                    continue

                st.markdown(
                    f'<div class="result-card"><div class="segment-name">Segmento {int(row["segment_id"])} — {row["segment_name"]}</div></div>',
                    unsafe_allow_html=True,
                )
                st.write("**Patrón dominante**")
                st.write(row["dominant_pattern"])
                st.write("**Objetivo comercial**")
                st.write(row["business_objective"])
                st.write("**Acción recomendada**")
                st.write(row["recommended_action"])
                st.write("**KPIs sugeridos**")
                st.write(row["suggested_kpi"])

                distance_col, threshold_col = st.columns(2)
                distance_col.metric(
                    "Distancia al centroide",
                    f"{row['distance_to_centroid']:.3f}",
                )
                threshold_col.metric(
                    "Umbral habitual (P95)",
                    f"{row['distance_threshold_p95']:.3f}",
                )
                if "alejado" in row["assignment_assessment"].lower():
                    st.warning(row["assignment_assessment"])
                else:
                    st.success(row["assignment_assessment"])

        csv_output = results.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descargar resultados en CSV",
            data=csv_output,
            file_name="resultados_segmentacion.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("Variables agregadas calculadas"):
            st.dataframe(features, use_container_width=True)

    except InputValidationError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("Se produjo un error inesperado durante la segmentación.")
        st.exception(exc)
