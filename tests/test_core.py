import pandas as pd

from segmentation_core import prepare_transactions, transactions_to_customer_features


def test_feature_construction():
    rows = []
    for i in range(35):
        rows.append(
            {
                "cc_num": "TEST_001",
                "trans_date_trans_time": f"2020-01-{(i % 28) + 1:02d} 10:00:00",
                "amt": 20 + i,
                "category": f"cat_{i % 4}",
                "merchant": f"merchant_{i % 8}",
            }
        )
    df = pd.DataFrame(rows)
    clean, audit = prepare_transactions(df)
    features = transactions_to_customer_features(clean, pd.Timestamp("2020-02-15"))
    assert audit["customers"] == 1
    assert int(features.loc[0, "frequency"]) == 35
    assert int(features.loc[0, "category_nunique"]) == 4
