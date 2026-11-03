import numpy as np
import pandas as pd

from cheatdetect.data import Log1pSkewed, find_skewed_features


def test_find_skewed_features_selects_only_nonnegative_right_skew():
    train = pd.DataFrame(
        {
            "right_skewed": [0] * 9 + [100],
            "has_negative_value": [-100] + [0] * 9,
            "balanced": list(range(10)),
        }
    )

    assert find_skewed_features(train) == ["right_skewed"]


def test_log_transform_changes_selected_column_and_reuses_selection():
    train = pd.DataFrame({"right_skewed": [0.0, 9.0], "other": [2.0, 4.0]})
    validation = pd.DataFrame({"right_skewed": [99.0], "other": [6.0]})
    transformer = Log1pSkewed(["right_skewed"]).fit(train)

    transformed_train = transformer.transform(train)
    transformed_validation = transformer.transform(validation)

    np.testing.assert_allclose(
        transformed_train["right_skewed"], np.log1p([0.0, 9.0])
    )
    np.testing.assert_allclose(
        transformed_validation["right_skewed"], np.log1p([99.0])
    )
    assert transformed_train["other"].tolist() == [2.0, 4.0]
    assert transformed_validation["other"].tolist() == [6.0]
    assert train["right_skewed"].tolist() == [0.0, 9.0]
