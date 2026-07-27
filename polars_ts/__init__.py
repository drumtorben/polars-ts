from pathlib import Path
from typing import Any

import polars as pl
import polars_ts_rs as _rs_mod
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

from polars_ts._distance_dispatch import (
    compute_pairwise_ddtw,
    compute_pairwise_dtw,
    compute_pairwise_dtw_multi,
    compute_pairwise_edr,
    compute_pairwise_erp,
    compute_pairwise_frechet,
    compute_pairwise_lcss,
    compute_pairwise_msm,
    compute_pairwise_msm_multi,
    compute_pairwise_sbd,
    compute_pairwise_twe,
    compute_pairwise_wdtw,
)
from polars_ts.distance import compute_pairwise_distance

PLUGIN_PATH = Path(_rs_mod.__file__).parent


def mann_kendall(expr: IntoExpr) -> pl.Expr:
    """Mann-Kendall test for expression."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="mann_kendall",
        args=expr,
        is_elementwise=False,
    )


def sens_slope(expr: IntoExpr) -> pl.Expr:
    """Sen's slope estimator (median of pairwise slopes)."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="sens_slope",
        args=expr,
        is_elementwise=False,
    )


# ---------------------------------------------------------------------------
# Lazy-import registry: name -> (module_path, attribute_name)
#
# Adding a new public name only requires one line here — no if-chains,
# no merge conflicts.
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # --- Metrics ---
    "Metrics": ("polars_ts.metrics", "Metrics"),
    "mae": ("polars_ts.metrics.forecast", "mae"),
    "rmse": ("polars_ts.metrics.forecast", "rmse"),
    "mape": ("polars_ts.metrics.forecast", "mape"),
    "smape": ("polars_ts.metrics.forecast", "smape"),
    "mase": ("polars_ts.metrics.forecast", "mase"),
    "crps": ("polars_ts.metrics.forecast", "crps"),
    # --- Decomposition ---
    "fourier_decomposition": ("polars_ts.decomposition.fourier_decomposition", "fourier_decomposition"),
    "seasonal_decomposition": ("polars_ts.decomposition.seasonal_decomposition", "seasonal_decomposition"),
    "seasonal_decompose_features": (
        "polars_ts.decomposition.seasonal_decompose_features",
        "seasonal_decompose_features",
    ),
    # --- Changepoint ---
    "cusum": ("polars_ts.changepoint.cusum", "cusum"),
    "pelt": ("polars_ts.changepoint.pelt", "pelt"),
    "bocpd": ("polars_ts.changepoint.bocpd", "bocpd"),
    "regime_detect": ("polars_ts.changepoint.regime", "regime_detect"),
    # --- Clustering ---
    "kmedoids": ("polars_ts.clustering.kmedoids", "kmedoids"),
    "TimeSeriesKMedoids": ("polars_ts.clustering.kmedoids", "TimeSeriesKMedoids"),
    "KShape": ("polars_ts.clustering.kshape", "KShape"),
    "silhouette_score": ("polars_ts.clustering.evaluation", "silhouette_score"),
    "silhouette_samples": ("polars_ts.clustering.evaluation", "silhouette_samples"),
    "davies_bouldin_score": ("polars_ts.clustering.evaluation", "davies_bouldin_score"),
    "calinski_harabasz_score": ("polars_ts.clustering.evaluation", "calinski_harabasz_score"),
    "hdbscan_cluster": ("polars_ts.clustering.density", "hdbscan_cluster"),
    "dbscan_cluster": ("polars_ts.clustering.density", "dbscan_cluster"),
    "spectral_cluster": ("polars_ts.clustering.spectral", "spectral_cluster"),
    "auto_cluster": ("polars_ts.clustering.auto", "auto_cluster"),
    "shapelet_cluster": ("polars_ts.clustering.shapelets", "shapelet_cluster"),
    "UShapeletClusterer": ("polars_ts.clustering.shapelets", "UShapeletClusterer"),
    "clara": ("polars_ts.clustering.scalable", "clara"),
    "clarans": ("polars_ts.clustering.scalable", "clarans"),
    "kmeans_dba": ("polars_ts.clustering.kmeans", "kmeans_dba"),
    "TimeSeriesKMeans": ("polars_ts.clustering.kmeans", "TimeSeriesKMeans"),
    "agglomerative_cluster": ("polars_ts.clustering.hierarchical", "agglomerative_cluster"),
    "ContrastiveClusterer": ("polars_ts.clustering.contrastive", "ContrastiveClusterer"),
    "contrastive_cluster": ("polars_ts.clustering.contrastive", "contrastive_cluster"),
    "DECClusterer": ("polars_ts.clustering.deep_cluster", "DECClusterer"),
    "IDECClusterer": ("polars_ts.clustering.deep_cluster", "IDECClusterer"),
    "dec_cluster": ("polars_ts.clustering.deep_cluster", "dec_cluster"),
    "idec_cluster": ("polars_ts.clustering.deep_cluster", "idec_cluster"),
    "KASBAClusterer": ("polars_ts.clustering.kasba", "KASBAClusterer"),
    "kasba": ("polars_ts.clustering.kasba", "kasba"),
    # --- Classification ---
    "knn_classify": ("polars_ts.classification.knn", "knn_classify"),
    "TimeSeriesKNNClassifier": ("polars_ts.classification.knn", "TimeSeriesKNNClassifier"),
    "KShapeClassifier": ("polars_ts.classification.kshape_classifier", "KShapeClassifier"),
    # --- Feature engineering ---
    "lag_features": ("polars_ts.features", "lag_features"),
    "covariate_lag_features": ("polars_ts.features", "covariate_lag_features"),
    "rolling_features": ("polars_ts.features", "rolling_features"),
    "calendar_features": ("polars_ts.features", "calendar_features"),
    "fourier_features": ("polars_ts.features", "fourier_features"),
    "rocket_features": ("polars_ts.features", "rocket_features"),
    "minirocket_features": ("polars_ts.features", "minirocket_features"),
    "target_encode": ("polars_ts.features.advanced", "target_encode"),
    "holiday_features": ("polars_ts.features.advanced", "holiday_features"),
    "interaction_features": ("polars_ts.features.advanced", "interaction_features"),
    "time_embeddings": ("polars_ts.features.advanced", "time_embeddings"),
    # --- Target transforms ---
    "log_transform": ("polars_ts.transforms", "log_transform"),
    "inverse_log_transform": ("polars_ts.transforms", "inverse_log_transform"),
    "boxcox_transform": ("polars_ts.transforms", "boxcox_transform"),
    "inverse_boxcox_transform": ("polars_ts.transforms", "inverse_boxcox_transform"),
    "difference": ("polars_ts.transforms", "difference"),
    "undifference": ("polars_ts.transforms", "undifference"),
    # --- Validation ---
    "expanding_window_cv": ("polars_ts.validation", "expanding_window_cv"),
    "sliding_window_cv": ("polars_ts.validation", "sliding_window_cv"),
    "rolling_origin_cv": ("polars_ts.validation", "rolling_origin_cv"),
    # --- Backtesting ---
    "backtest": ("polars_ts.backtesting", "backtest"),
    "compare_models": ("polars_ts.backtesting", "compare_models"),
    # --- Models & forecasting ---
    "SCUM": ("polars_ts.models", "SCUM"),
    "naive_forecast": ("polars_ts.models", "naive_forecast"),
    "seasonal_naive_forecast": ("polars_ts.models", "seasonal_naive_forecast"),
    "moving_average_forecast": ("polars_ts.models", "moving_average_forecast"),
    "fft_forecast": ("polars_ts.models", "fft_forecast"),
    "RecursiveForecaster": ("polars_ts.models", "RecursiveForecaster"),
    "DirectForecaster": ("polars_ts.models", "DirectForecaster"),
    "ses_forecast": ("polars_ts.models", "ses_forecast"),
    "holt_forecast": ("polars_ts.models", "holt_forecast"),
    "holt_winters_forecast": ("polars_ts.models", "holt_winters_forecast"),
    "arima_fit": ("polars_ts.models", "arima_fit"),
    "arima_forecast": ("polars_ts.models", "arima_forecast"),
    "auto_arima": ("polars_ts.models", "auto_arima"),
    "ForecastPipeline": ("polars_ts.pipeline", "ForecastPipeline"),
    "GlobalForecaster": ("polars_ts.global_model", "GlobalForecaster"),
    # --- Ensembles ---
    "WeightedEnsemble": ("polars_ts.ensemble", "WeightedEnsemble"),
    "StackingForecaster": ("polars_ts.ensemble", "StackingForecaster"),
    # --- Probabilistic ---
    "QuantileRegressor": ("polars_ts.probabilistic", "QuantileRegressor"),
    "conformal_interval": ("polars_ts.probabilistic", "conformal_interval"),
    "EnbPI": ("polars_ts.probabilistic", "EnbPI"),
    # --- Volatility ---
    "garch_fit": ("polars_ts.volatility", "garch_fit"),
    "garch_forecast": ("polars_ts.volatility", "garch_forecast"),
    "GARCHResult": ("polars_ts.volatility", "GARCHResult"),
    # --- VAR ---
    "var_fit": ("polars_ts.var_model", "var_fit"),
    "var_forecast": ("polars_ts.var_model", "var_forecast"),
    "granger_causality": ("polars_ts.var_model", "granger_causality"),
    "VARResult": ("polars_ts.var_model", "VARResult"),
    # --- Bayesian VAR ---
    "bayesian_var": ("polars_ts.bayesian_var", "bayesian_var"),
    "BayesianVAR": ("polars_ts.bayesian_var", "BayesianVAR"),
    "MinnesotaPrior": ("polars_ts.bayesian_var", "MinnesotaPrior"),
    "NormalWishartPrior": ("polars_ts.bayesian_var", "NormalWishartPrior"),
    "BayesianVARResult": ("polars_ts.bayesian_var", "BayesianVARResult"),
    # --- Reconciliation ---
    "reconcile": ("polars_ts.reconciliation", "reconcile"),
    # --- Adapters ---
    "to_neuralforecast": ("polars_ts.adapters", "to_neuralforecast"),
    "from_neuralforecast": ("polars_ts.adapters", "from_neuralforecast"),
    "to_pytorch_forecasting": ("polars_ts.adapters", "to_pytorch_forecasting"),
    "from_pytorch_forecasting": ("polars_ts.adapters", "from_pytorch_forecasting"),
    "to_hf_dataset": ("polars_ts.adapters", "to_hf_dataset"),
    "ForecastEnv": ("polars_ts.adapters", "ForecastEnv"),
    "to_chronos_embeddings": ("polars_ts.adapters", "to_chronos_embeddings"),
    "to_moment_embeddings": ("polars_ts.adapters", "to_moment_embeddings"),
    "foundation_forecast": ("polars_ts.adapters", "foundation_forecast"),
    "ChronosForecaster": ("polars_ts.adapters", "ChronosForecaster"),
    "TimesFMForecaster": ("polars_ts.adapters", "TimesFMForecaster"),
    "MoiraiForecaster": ("polars_ts.adapters", "MoiraiForecaster"),
    "TimeLLMForecaster": ("polars_ts.adapters", "TimeLLMForecaster"),
    "LLMPSForecaster": ("polars_ts.adapters", "LLMPSForecaster"),
    # --- Bias & calibration ---
    "bias_detect": ("polars_ts.bias", "bias_detect"),
    "bias_correct": ("polars_ts.bias", "bias_correct"),
    "calibration_table": ("polars_ts.calibration", "calibration_table"),
    "pit_histogram": ("polars_ts.calibration", "pit_histogram"),
    "reliability_diagram": ("polars_ts.calibration", "reliability_diagram"),
    # --- Feature importance ---
    "permutation_importance": ("polars_ts.importance", "permutation_importance"),
    # --- Anomaly detection ---
    "isolation_forest_detect": ("polars_ts.anomaly_forest", "isolation_forest_detect"),
    # --- Preprocessing ---
    "impute": ("polars_ts.imputation", "impute"),
    "detect_outliers": ("polars_ts.outliers", "detect_outliers"),
    "treat_outliers": ("polars_ts.outliers", "treat_outliers"),
    "resample": ("polars_ts.resampling", "resample"),
    # --- Diagnostics ---
    "acf": ("polars_ts.diagnostics", "acf"),
    "pacf": ("polars_ts.diagnostics", "pacf"),
    "ljung_box": ("polars_ts.diagnostics", "ljung_box"),
    # --- Bayesian ETS ---
    "bayesian_ets": ("polars_ts.models.bayesian_ets", "bayesian_ets"),
    "BayesianETS": ("polars_ts.models.bayesian_ets", "BayesianETS"),
    "ETSPriors": ("polars_ts.models.bayesian_ets", "ETSPriors"),
    # --- Causal Inference ---
    "CausalImpact": ("polars_ts.causal.causal_impact", "CausalImpact"),
    "causal_impact": ("polars_ts.causal.causal_impact", "causal_impact"),
    "CausalImpactResult": ("polars_ts.causal.causal_impact", "CausalImpactResult"),
    "SyntheticControl": ("polars_ts.causal.synthetic_control", "SyntheticControl"),
    "synthetic_control": ("polars_ts.causal.synthetic_control", "synthetic_control"),
    "SyntheticControlResult": ("polars_ts.causal.synthetic_control", "SyntheticControlResult"),
    # --- Agents ---
    "TimeSeriesScientist": ("polars_ts.agents", "TimeSeriesScientist"),
    "ScientistResult": ("polars_ts.agents", "ScientistResult"),
    "CuratorAgent": ("polars_ts.agents", "CuratorAgent"),
    "PlannerAgent": ("polars_ts.agents", "PlannerAgent"),
    "ForecasterAgent": ("polars_ts.agents", "ForecasterAgent"),
    "ReporterAgent": ("polars_ts.agents", "ReporterAgent"),
    # --- Multivariate DL ---
    "MultivariatePatchTST": ("polars_ts.dl", "MultivariatePatchTST"),
    "iTransformerForecaster": ("polars_ts.dl", "iTransformerForecaster"),
    # --- Anomaly detection agents ---
    "AnomalyEnv": ("polars_ts.anomaly_agents", "AnomalyEnv"),
    "AnomalyOrchestrator": ("polars_ts.anomaly_agents", "AnomalyOrchestrator"),
    "AnomalyResult": ("polars_ts.anomaly_agents", "AnomalyResult"),
    "ZScoreAgent": ("polars_ts.anomaly_agents", "ZScoreAgent"),
    "RollingStdAgent": ("polars_ts.anomaly_agents", "RollingStdAgent"),
    "MADAgent": ("polars_ts.anomaly_agents", "MADAgent"),
    "ConsensusAgent": ("polars_ts.anomaly_agents", "ConsensusAgent"),
    # --- Supply chain demand-sensing agents ---
    "SupplyChainOrchestrator": ("polars_ts.supply_chain_agents", "SupplyChainOrchestrator"),
    "SupplyChainResult": ("polars_ts.supply_chain_agents", "SupplyChainResult"),
    "DemandSensingAgent": ("polars_ts.supply_chain_agents", "DemandSensingAgent"),
    "PromotionEffectAgent": ("polars_ts.supply_chain_agents", "PromotionEffectAgent"),
    "InventoryAgent": ("polars_ts.supply_chain_agents", "InventoryAgent"),
    "EchelonCoordinatorAgent": ("polars_ts.supply_chain_agents", "EchelonCoordinatorAgent"),
    # --- Energy/demand forecasting agents ---
    "GridHierarchy": ("polars_ts.energy_agents", "GridHierarchy"),
    "EnergyGridOrchestrator": ("polars_ts.energy_agents", "EnergyGridOrchestrator"),
    "EnergyForecastResult": ("polars_ts.energy_agents", "EnergyForecastResult"),
    "DemandForecastAgent": ("polars_ts.energy_agents", "DemandForecastAgent"),
    "WeatherContextAgent": ("polars_ts.energy_agents", "WeatherContextAgent"),
    "RenewableAgent": ("polars_ts.energy_agents", "RenewableAgent"),
    "DemandResponseAgent": ("polars_ts.energy_agents", "DemandResponseAgent"),
    # --- Industrial IoT predictive maintenance agents ---
    "MachineEnv": ("polars_ts.iiot_agents", "MachineEnv"),
    "MaintenanceOrchestrator": ("polars_ts.iiot_agents", "MaintenanceOrchestrator"),
    "MaintenanceResult": ("polars_ts.iiot_agents", "MaintenanceResult"),
    "SpectralFeatureAgent": ("polars_ts.iiot_agents", "SpectralFeatureAgent"),
    "HealthIndexAgent": ("polars_ts.iiot_agents", "HealthIndexAgent"),
    "RULEstimator": ("polars_ts.iiot_agents", "RULEstimator"),
    "MaintenanceSchedulerAgent": ("polars_ts.iiot_agents", "MaintenanceSchedulerAgent"),
    # --- Healthcare agents (clinical decision support) ---
    "ClinicalEnv": ("polars_ts.healthcare_agents", "ClinicalEnv"),
    "ClinicalOrchestrator": ("polars_ts.healthcare_agents", "ClinicalOrchestrator"),
    "ClinicalResult": ("polars_ts.healthcare_agents", "ClinicalResult"),
    "SepsisWarningAgent": ("polars_ts.healthcare_agents", "SepsisWarningAgent"),
    "VitalMonitorAgent": ("polars_ts.healthcare_agents", "VitalMonitorAgent"),
    "EscalationAgent": ("polars_ts.healthcare_agents", "EscalationAgent"),
    "TreatmentAgent": ("polars_ts.healthcare_agents", "TreatmentAgent"),
    "federated_average": ("polars_ts.healthcare_agents", "federated_average"),
    # --- Multi-agent RL ---
    "PortfolioEnv": ("polars_ts.marl", "PortfolioEnv"),
    "MARLOrchestrator": ("polars_ts.marl", "MARLOrchestrator"),
    "MARLResult": ("polars_ts.marl", "MARLResult"),
    "RiskAgent": ("polars_ts.marl", "RiskAgent"),
    "ReturnAgent": ("polars_ts.marl", "ReturnAgent"),
    "AllocationAgent": ("polars_ts.marl", "AllocationAgent"),
    # --- Streaming / Online Learning ---
    "StreamingETS": ("polars_ts.streaming", "StreamingETS"),
    "StreamingKalmanFilter": ("polars_ts.streaming", "StreamingKalmanFilter"),
    "StreamingGlobalForecaster": ("polars_ts.streaming", "StreamingGlobalForecaster"),
    "SlidingWindowManager": ("polars_ts.streaming", "SlidingWindowManager"),
    # --- Registry / Experiment Tracking ---
    "ModelRegistry": ("polars_ts.registry", "ModelRegistry"),
    "Experiment": ("polars_ts.registry", "Experiment"),
    "Run": ("polars_ts.registry", "Run"),
    # --- Bayesian ---
    "KalmanFilter": ("polars_ts.bayesian", "KalmanFilter"),
    "kalman_filter": ("polars_ts.bayesian", "kalman_filter"),
    "UnscentedKalmanFilter": ("polars_ts.bayesian", "UnscentedKalmanFilter"),
    "EnsembleKalmanFilter": ("polars_ts.bayesian", "EnsembleKalmanFilter"),
    "BSTS": ("polars_ts.bayesian", "BSTS"),
    "bsts_fit": ("polars_ts.bayesian", "bsts_fit"),
    "bsts_forecast": ("polars_ts.bayesian", "bsts_forecast"),
    "GaussianProcessTS": ("polars_ts.bayesian", "GaussianProcessTS"),
    "gp_forecast": ("polars_ts.bayesian", "gp_forecast"),
    "MCMCForecaster": ("polars_ts.bayesian", "MCMCForecaster"),
    "mcmc_forecast": ("polars_ts.bayesian", "mcmc_forecast"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib

        mod_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'polars_ts' has no attribute {name!r}")


__all__ = [
    "compute_pairwise_distance",
    "compute_pairwise_dtw",
    "compute_pairwise_ddtw",
    "compute_pairwise_wdtw",
    "compute_pairwise_msm",
    "compute_pairwise_dtw_multi",
    "compute_pairwise_msm_multi",
    "compute_pairwise_erp",
    "compute_pairwise_lcss",
    "compute_pairwise_twe",
    "compute_pairwise_sbd",
    "compute_pairwise_frechet",
    "compute_pairwise_edr",
    "mann_kendall",
    "sens_slope",
    *_LAZY_IMPORTS.keys(),
]
