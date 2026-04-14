import polars as pl
import pytest

from polars_ts.classification.knn import knn_classify
from polars_ts.clustering.kmedoids import kmedoids


@pytest.fixture
def train_data():
    """Training data with three distinct classes."""
    return pl.DataFrame(
        {
            "unique_id": (
                ["train_A1"] * 5 + ["train_A2"] * 5
                + ["train_B1"] * 5 + ["train_B2"] * 5
                + ["train_C1"] * 5 + ["train_C2"] * 5
            ),
            "y": (
                [1.0, 1.0, 1.0, 1.0, 1.0]  # class A - flat low
                + [1.0, 1.0, 1.0, 1.0, 1.1]
                + [5.0, 5.0, 5.0, 5.0, 5.0]  # class B - flat high
                + [5.0, 5.0, 5.0, 5.0, 5.1]
                + [1.0, 5.0, 1.0, 5.0, 1.0]  # class C - oscillating
                + [1.0, 5.0, 1.0, 5.0, 1.1]
            ),
            "label": (
                ["A"] * 5 + ["A"] * 5
                + ["B"] * 5 + ["B"] * 5
                + ["C"] * 5 + ["C"] * 5
            ),
        }
    )


@pytest.fixture
def test_data():
    """Test data matching the three classes."""
    return pl.DataFrame(
        {
            "unique_id": ["test_1"] * 5 + ["test_2"] * 5 + ["test_3"] * 5,
            "y": (
                [1.0, 1.0, 1.0, 1.0, 1.05]  # should be class A
                + [5.0, 5.0, 5.0, 5.0, 5.05]  # should be class B
                + [1.0, 5.0, 1.0, 5.0, 1.05]  # should be class C
            ),
        }
    )


class TestKnnOutput:
    def test_returns_correct_columns(self, train_data, test_data):
        result = knn_classify(train_data, test_data, k=1)
        assert result.columns == ["unique_id", "predicted_label"]

    def test_all_test_ids_classified(self, train_data, test_data):
        result = knn_classify(train_data, test_data, k=1)
        test_ids = set(test_data["unique_id"].unique().to_list())
        result_ids = set(result["unique_id"].to_list())
        assert test_ids == result_ids

    def test_label_dtype_preserved(self, train_data, test_data):
        result = knn_classify(train_data, test_data, k=1)
        assert result["predicted_label"].dtype == train_data["label"].dtype


class TestKnnCorrectness:
    def test_perfect_classification_k1(self, train_data, test_data):
        result = knn_classify(train_data, test_data, k=1)
        preds = dict(zip(result["unique_id"].to_list(), result["predicted_label"].to_list()))
        assert preds["test_1"] == "A"
        assert preds["test_2"] == "B"
        assert preds["test_3"] == "C"

    def test_perfect_classification_k2(self, train_data, test_data):
        result = knn_classify(train_data, test_data, k=2)
        preds = dict(zip(result["unique_id"].to_list(), result["predicted_label"].to_list()))
        assert preds["test_1"] == "A"
        assert preds["test_2"] == "B"
        assert preds["test_3"] == "C"

    def test_single_class_train(self, test_data):
        train = pl.DataFrame(
            {
                "unique_id": ["t1"] * 5 + ["t2"] * 5,
                "y": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
                "label": ["X"] * 10,
            }
        )
        result = knn_classify(train, test_data, k=1)
        assert all(label == "X" for label in result["predicted_label"].to_list())


class TestKnnEdgeCases:
    def test_k1_returns_nearest(self, train_data, test_data):
        result = knn_classify(train_data, test_data, k=1)
        assert result.shape[0] == 3

    def test_missing_label_col_raises(self, test_data):
        train_no_label = pl.DataFrame(
            {"unique_id": ["A"] * 3, "y": [1.0, 2.0, 3.0]}
        )
        with pytest.raises(ValueError, match="label column"):
            knn_classify(train_no_label, test_data, k=1)

    def test_int_ids(self):
        train = pl.DataFrame(
            {
                "unique_id": [1] * 4 + [2] * 4,
                "y": [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0],
                "label": ["low"] * 4 + ["high"] * 4,
            }
        )
        test = pl.DataFrame(
            {"unique_id": [3] * 4, "y": [1.0, 1.0, 1.0, 1.1]}
        )
        result = knn_classify(train, test, k=1)
        assert result["unique_id"].dtype == pl.Int64
        assert result["predicted_label"].to_list() == ["low"]


class TestKnnRepeatedValues:
    def test_series_with_repeated_values_preserves_length(self):
        """Train series with repeated y-values must not have rows dropped."""
        train = pl.DataFrame(
            {
                "unique_id": ["A"] * 6 + ["B"] * 6,
                "y": [0.0, 0.0, 0.0, 0.0, 0.0, 10.0,  # A: flat then spike
                      10.0, 10.0, 10.0, 10.0, 10.0, 0.0],  # B: flat then drop
                "label": ["spike"] * 6 + ["drop"] * 6,
            }
        )
        # Test series matches A's pattern (flat then spike)
        test = pl.DataFrame(
            {"unique_id": ["X"] * 6, "y": [0.0, 0.0, 0.0, 0.0, 0.0, 9.5]}
        )
        result = knn_classify(train, test, k=1)
        preds = dict(zip(result["unique_id"].to_list(), result["predicted_label"].to_list()))
        assert preds["X"] == "spike"

    def test_custom_column_names(self):
        """Custom id_col and target_col should work correctly."""
        train = pl.DataFrame(
            {
                "series_name": ["A"] * 4 + ["B"] * 4,
                "value": [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0],
                "class": ["low"] * 4 + ["high"] * 4,
            }
        )
        test = pl.DataFrame(
            {"series_name": ["X"] * 4, "value": [5.0, 5.0, 5.0, 4.9]}
        )
        result = knn_classify(
            train, test, k=1,
            id_col="series_name", target_col="value", label_col="class",
        )
        assert result.columns == ["series_name", "predicted_label"]
        assert result["predicted_label"].to_list() == ["high"]

    def test_kmedoids_custom_columns(self):
        """Custom id_col and target_col should work for kmedoids too."""
        df = pl.DataFrame(
            {
                "ts_id": ["A"] * 4 + ["B"] * 4,
                "val": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.1],
            }
        )
        result = kmedoids(df, k=1, id_col="ts_id", target_col="val")
        assert result.columns == ["ts_id", "cluster"]


class TestKnnMetrics:
    def test_with_erp(self, train_data, test_data):
        result = knn_classify(train_data, test_data, k=1, method="erp")
        assert result.shape[0] == 3

    def test_with_msm(self, train_data, test_data):
        result = knn_classify(train_data, test_data, k=1, method="msm")
        assert result.shape[0] == 3

    def test_with_wdtw(self, train_data, test_data):
        result = knn_classify(train_data, test_data, k=1, method="wdtw", g=0.1)
        assert result.shape[0] == 3
