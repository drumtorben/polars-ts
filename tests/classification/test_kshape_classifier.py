import polars as pl
import pytest

from polars_ts.classification import KShapeClassifier


@pytest.fixture
def train_data():
    """Training data with two distinct shape classes."""
    return pl.DataFrame(
        {
            "unique_id": ["A"] * 8 + ["B"] * 8 + ["C"] * 8 + ["D"] * 8,
            "y": (
                [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]
                + [0.0, 0.9, 0.0, -0.9, 0.0, 0.9, 0.0, -0.9]
                + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                + [1.1, 0.9, 1.0, 1.0, 1.1, 0.9, 1.0, 1.0]
            ),
            "label": ["wave"] * 8 + ["wave"] * 8 + ["flat"] * 8 + ["flat"] * 8,
        }
    )


@pytest.fixture
def test_data():
    return pl.DataFrame(
        {
            "unique_id": ["X"] * 8 + ["Y"] * 8,
            "y": ([0.0, 0.8, 0.0, -0.8, 0.0, 0.8, 0.0, -0.8] + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        }
    )


class TestKShapeClassifier:
    def test_fit_predict(self, train_data, test_data):
        clf = KShapeClassifier(n_centroids_per_class=1)
        clf.fit(train_data, label_col="label")
        result = clf.predict(test_data)

        assert "unique_id" in result.columns
        assert "predicted_label" in result.columns
        assert result.shape[0] == 2

    def test_classification_accuracy(self, train_data, test_data):
        clf = KShapeClassifier(n_centroids_per_class=1)
        clf.fit(train_data, label_col="label")
        result = clf.predict(test_data)

        preds = dict(zip(result["unique_id"].to_list(), result["predicted_label"].to_list(), strict=False))
        assert preds["X"] == "wave"
        assert preds["Y"] == "flat"

    def test_multi_centroids_per_class(self):
        """Test the n_centroids_per_class > 1 path (lines 78-81)."""
        train = pl.DataFrame(
            {
                "unique_id": (["A"] * 8 + ["B"] * 8 + ["C"] * 8 + ["D"] * 8 + ["E"] * 8 + ["F"] * 8),
                "y": (
                    [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]
                    + [0.0, 0.9, 0.0, -0.9, 0.0, 0.9, 0.0, -0.9]
                    + [0.0, 0.8, 0.0, -0.8, 0.0, 0.8, 0.0, -0.8]
                    + [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
                    + [4.9, 5.1, 5.0, 5.0, 4.9, 5.1, 5.0, 5.0]
                    + [5.1, 4.9, 5.0, 5.0, 5.1, 4.9, 5.0, 5.0]
                ),
                "label": (["wave"] * 8 + ["wave"] * 8 + ["wave"] * 8 + ["flat"] * 8 + ["flat"] * 8 + ["flat"] * 8),
            }
        )
        test = pl.DataFrame(
            {
                "unique_id": ["X"] * 8,
                "y": [0.0, 0.7, 0.0, -0.7, 0.0, 0.7, 0.0, -0.7],
            }
        )
        clf = KShapeClassifier(n_centroids_per_class=2)
        clf.fit(train, label_col="label")
        result = clf.predict(test)
        assert result.shape[0] == 1
        # Should have 2 centroids per class = 4 total
        assert len(clf._centroids) == 4

    def test_predict_before_fit_raises(self, test_data):
        clf = KShapeClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(test_data)
