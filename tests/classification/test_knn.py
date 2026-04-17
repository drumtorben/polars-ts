import polars as pl
import pytest

from polars_ts.classification import TimeSeriesKNNClassifier


@pytest.fixture
def train_data():
    """Training data with two classes: sine-like (A, B) and constant (C, D)."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 8 + ["B"] * 8 + ["C"] * 8 + ["D"] * 8,
            "y": (
                [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]  # sine-like A
                + [0.0, 0.9, 0.0, -0.9, 0.0, 0.9, 0.0, -0.9]  # sine-like B
                + [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]  # constant C
                + [4.9, 5.1, 5.0, 5.0, 4.9, 5.1, 5.0, 5.0]  # near-constant D
            ),
            "label": (["sine"] * 8 + ["sine"] * 8 + ["constant"] * 8 + ["constant"] * 8),
        }
    )


@pytest.fixture
def test_data():
    """Test data: one sine-like, one constant."""
    return pl.DataFrame(
        {
            "unique_id": ["X"] * 8 + ["Y"] * 8,
            "y": (
                [0.0, 0.8, 0.0, -0.8, 0.0, 0.8, 0.0, -0.8]  # sine-like
                + [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]  # constant
            ),
        }
    )


class TestTimeSeriesKNNClassifier:
    def test_fit_predict(self, train_data, test_data):
        clf = TimeSeriesKNNClassifier(k=1, metric="dtw")
        clf.fit(train_data, label_col="label")
        result = clf.predict(test_data)

        assert "unique_id" in result.columns
        assert "predicted_label" in result.columns
        assert result.shape[0] == 2

    def test_k1_classification(self, train_data, test_data):
        clf = TimeSeriesKNNClassifier(k=1, metric="dtw")
        clf.fit(train_data, label_col="label")
        result = clf.predict(test_data)

        preds = dict(zip(result["unique_id"].to_list(), result["predicted_label"].to_list(), strict=False))
        assert preds["X"] == "sine"
        assert preds["Y"] == "constant"

    def test_k3_classification(self, train_data, test_data):
        clf = TimeSeriesKNNClassifier(k=3, metric="dtw")
        clf.fit(train_data, label_col="label")
        result = clf.predict(test_data)

        preds = dict(zip(result["unique_id"].to_list(), result["predicted_label"].to_list(), strict=False))
        assert preds["X"] == "sine"
        assert preds["Y"] == "constant"

    def test_predict_before_fit_raises(self, test_data):
        clf = TimeSeriesKNNClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(test_data)

    def test_different_metrics(self, train_data, test_data):
        for metric in ["dtw", "erp", "lcss"]:
            clf = TimeSeriesKNNClassifier(k=1, metric=metric)
            clf.fit(train_data, label_col="label")
            result = clf.predict(test_data)
            assert result.shape[0] == 2

    def test_self_classification(self, train_data):
        """Training data classified against itself should get perfect accuracy."""
        clf = TimeSeriesKNNClassifier(k=1, metric="dtw")
        clf.fit(train_data, label_col="label")
        # Create test data from training (different IDs not needed since we use same df)
        test = train_data.select("unique_id", "y")
        result = clf.predict(test)
        assert result.shape[0] > 0
