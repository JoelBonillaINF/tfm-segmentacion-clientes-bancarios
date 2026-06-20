from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

REQUIRED_TRANSACTION_COLUMNS = [
    "cc_num",
    "trans_date_trans_time",
    "amt",
    "category",
    "merchant",
]

PERIOD_COLUMNS = [
    "txn_pct_morning",
    "txn_pct_afternoon",
    "txn_pct_night",
    "txn_pct_early_morning",
]


class InputValidationError(ValueError):
    """Error de validación legible para la interfaz."""


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Carga el paquete persistido del modelo de segmentación."""
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"No se encontró el paquete del modelo en: {bundle_path.resolve()}"
        )
    return joblib.load(bundle_path)


def _time_period(hour: int) -> str:
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 24:
        return "night"
    return "early_morning"


def prepare_transactions(transactions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Valida y normaliza un historial transaccional cargado por el usuario."""
    if transactions is None or transactions.empty:
        raise InputValidationError("El archivo no contiene transacciones.")

    df = transactions.copy()
    df.columns = [str(col).strip() for col in df.columns]

    missing = [col for col in REQUIRED_TRANSACTION_COLUMNS if col not in df.columns]
    if missing:
        raise InputValidationError(
            "Faltan columnas requeridas: " + ", ".join(missing)
        )

    original_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        df = df.drop_duplicates().copy()

    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"], errors="coerce"
    )
    invalid_dates = int(df["trans_date_trans_time"].isna().sum())
    if invalid_dates:
        raise InputValidationError(
            f"Se encontraron {invalid_dates} fechas no válidas en trans_date_trans_time."
        )

    df["amt"] = pd.to_numeric(df["amt"], errors="coerce")
    invalid_amounts = int(df["amt"].isna().sum())
    if invalid_amounts:
        raise InputValidationError(
            f"Se encontraron {invalid_amounts} montos no numéricos en amt."
        )

    null_counts = df[REQUIRED_TRANSACTION_COLUMNS].isna().sum()
    null_columns = null_counts[null_counts > 0]
    if not null_columns.empty:
        detail = ", ".join(f"{col}: {int(value)}" for col, value in null_columns.items())
        raise InputValidationError(
            "Existen valores ausentes en columnas obligatorias: " + detail
        )

    if (df["amt"] < 0).any():
        negative_count = int((df["amt"] < 0).sum())
        raise InputValidationError(
            f"Se encontraron {negative_count} montos negativos. Deben corregirse antes de segmentar."
        )

    df["cc_num"] = df["cc_num"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["merchant"] = df["merchant"].astype(str).str.strip()

    empty_ids = int((df["cc_num"] == "").sum())
    if empty_ids:
        raise InputValidationError(
            f"Se encontraron {empty_ids} identificadores de cliente vacíos."
        )

    audit = {
        "original_rows": original_rows,
        "duplicate_rows_removed": duplicate_rows,
        "clean_rows": len(df),
        "customers": int(df["cc_num"].nunique()),
    }
    return df.sort_values("trans_date_trans_time").reset_index(drop=True), audit


def transactions_to_customer_features(
    transactions: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Replica la ingeniería de variables del notebook para uno o varios clientes."""
    df, _ = prepare_transactions(transactions)
    reference_ts = pd.Timestamp(reference_date)

    if reference_ts < df["trans_date_trans_time"].max():
        raise InputValidationError(
            "La fecha de corte no puede ser anterior a la última transacción cargada."
        )

    df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
    df["transaction_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
    df["is_weekend"] = df["transaction_dayofweek"].isin([5, 6]).astype(int)
    df["time_period"] = df["transaction_hour"].apply(_time_period)

    client_base = (
        df.groupby("cc_num")
        .agg(
            first_transaction=("trans_date_trans_time", "min"),
            last_transaction=("trans_date_trans_time", "max"),
            frequency=("amt", "size"),
            monetary_total=("amt", "sum"),
            ticket_mean=("amt", "mean"),
            ticket_median=("amt", "median"),
            ticket_std=("amt", "std"),
            ticket_min=("amt", "min"),
            ticket_max=("amt", "max"),
            category_nunique=("category", "nunique"),
            merchant_nunique=("merchant", "nunique"),
            weekend_txn_pct=("is_weekend", "mean"),
        )
        .reset_index()
    )

    client_base["recency_days"] = (
        reference_ts - client_base["last_transaction"]
    ).dt.days
    client_base["customer_lifetime_days"] = (
        client_base["last_transaction"] - client_base["first_transaction"]
    ).dt.days + 1
    client_base["ticket_std"] = client_base["ticket_std"].fillna(0.0)

    period_pivot = df.pivot_table(
        index="cc_num",
        columns="time_period",
        values="amt",
        aggfunc="size",
        fill_value=0,
    )
    period_pivot = period_pivot.div(period_pivot.sum(axis=1), axis=0)
    period_pivot.columns = [f"txn_pct_{col}" for col in period_pivot.columns]

    for col in PERIOD_COLUMNS:
        if col not in period_pivot.columns:
            period_pivot[col] = 0.0
    period_pivot = period_pivot[PERIOD_COLUMNS]

    client_category = (
        df.groupby(["cc_num", "category"])
        .agg(category_amount=("amt", "sum"), category_txn=("amt", "size"))
        .reset_index()
    )
    total_by_client = client_category.groupby("cc_num")["category_amount"].transform("sum")
    client_category["category_amount_share"] = np.where(
        total_by_client > 0,
        client_category["category_amount"] / total_by_client,
        0.0,
    )

    category_concentration = (
        client_category.sort_values(
            ["cc_num", "category_amount_share"], ascending=[True, False]
        )
        .groupby("cc_num")
        .agg(
            top_category_share=("category_amount_share", "max"),
            top3_category_share=("category_amount_share", lambda x: x.head(3).sum()),
        )
    )

    dominant_category = (
        client_category.sort_values(
            ["cc_num", "category_amount_share"], ascending=[True, False]
        )
        .groupby("cc_num")
        .first()[["category"]]
        .rename(columns={"category": "dominant_category"})
    )

    features = (
        client_base.set_index("cc_num")
        .join(period_pivot, how="left")
        .join(category_concentration, how="left")
        .join(dominant_category, how="left")
        .reset_index()
    )

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0.0)
    features["dominant_category"] = features["dominant_category"].fillna("sin_categoria")
    return features


def check_eligibility(features: pd.DataFrame, bundle: dict[str, Any]) -> pd.Series:
    """Determina qué clientes cuentan con historial suficiente."""
    return (
        (features["frequency"] >= int(bundle["min_transactions"]))
        & (
            features["customer_lifetime_days"]
            >= int(bundle["min_lifetime_days"])
        )
    )


def transform_customer_features(
    customer_features: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """Aplica exclusivamente transformaciones aprendidas durante el entrenamiento."""
    missing = [
        col for col in bundle["input_feature_order"] if col not in customer_features.columns
    ]
    if missing:
        raise InputValidationError(
            "No fue posible construir las variables requeridas: " + ", ".join(missing)
        )

    X = customer_features[bundle["input_feature_order"]].copy()
    X = X.drop(columns=bundle.get("excluded_columns", []), errors="ignore")

    for col, bounds in bundle["winsor_bounds"].items():
        if col in X.columns:
            X[col] = X[col].clip(
                lower=float(bounds["lower"]),
                upper=float(bounds["upper"]),
            )

    for col in bundle.get("log_columns", []):
        if col in X.columns:
            if (X[col] < 0).any():
                raise InputValidationError(
                    f"La variable {col} contiene valores negativos y no admite log1p."
                )
            X[col] = np.log1p(X[col])

    X = X.drop(columns=bundle.get("zero_variance_columns", []), errors="ignore")
    X = X[bundle["feature_order"]]

    scaled = bundle["scaler"].transform(X)
    return pd.DataFrame(scaled, columns=bundle["feature_order"], index=X.index)


def score_transactions(
    transactions: pd.DataFrame,
    reference_date: pd.Timestamp,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Valida, agrega, comprueba elegibilidad y asigna segmentos."""
    clean_transactions, audit = prepare_transactions(transactions)
    features = transactions_to_customer_features(clean_transactions, reference_date)
    eligible = check_eligibility(features, bundle)

    results = features[
        [
            "cc_num",
            "frequency",
            "customer_lifetime_days",
            "recency_days",
            "monetary_total",
            "ticket_mean",
            "merchant_nunique",
            "category_nunique",
            "dominant_category",
        ]
    ].copy()
    results["eligible"] = eligible.values
    results["status"] = np.where(
        results["eligible"],
        "Elegible para segmentación",
        "Historial insuficiente",
    )
    results["segment_id"] = pd.Series([pd.NA] * len(results), dtype="Int64")
    results["segment_name"] = "No asignado"
    results["dominant_pattern"] = ""
    results["business_objective"] = ""
    results["recommended_action"] = ""
    results["suggested_kpi"] = ""
    results["distance_to_centroid"] = np.nan
    results["distance_threshold_p95"] = np.nan
    results["assignment_assessment"] = "No aplica"

    if eligible.any():
        eligible_features = features.loc[eligible].reset_index(drop=True)
        X_scaled = transform_customer_features(eligible_features, bundle)
        model = bundle["model"]
        labels = model.predict(X_scaled)
        distances = model.transform(X_scaled)

        for position, segment_id in enumerate(labels):
            target_index = results.index[eligible][position]
            segment_id = int(segment_id)
            assigned_distance = float(distances[position, segment_id])
            threshold = float(bundle["distance_thresholds"][segment_id])
            business = bundle["segment_business_map"][segment_id]

            results.at[target_index, "segment_id"] = segment_id
            results.at[target_index, "segment_name"] = bundle["segment_name_map"][segment_id]
            results.at[target_index, "dominant_pattern"] = business["dominant_pattern"]
            results.at[target_index, "business_objective"] = business["business_objective"]
            results.at[target_index, "recommended_action"] = business["recommended_action"]
            results.at[target_index, "suggested_kpi"] = business["suggested_kpi"]
            results.at[target_index, "distance_to_centroid"] = assigned_distance
            results.at[target_index, "distance_threshold_p95"] = threshold
            results.at[target_index, "assignment_assessment"] = (
                "Dentro del rango habitual del segmento"
                if assigned_distance <= threshold
                else "Cliente alejado del centroide; revisar asignación"
            )

    return results, features, audit
