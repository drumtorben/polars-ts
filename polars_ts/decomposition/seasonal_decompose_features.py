from typing import Union, List,Literal, Optional 
import polars.selectors as cs 

from polars_ts.decomposition.seasonal_decomposition import seasonal_decomposition
from statsforecast.feature_engineering import mstl_decomposition
from statsforecast.models import MSTL  

from typing import List, Literal, Optional
import polars as pl

def seasonal_decompose_features(
    df: pl.DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
    ts_freq: str,
    seasonal_freqs: Optional[List[int]] = None,  # Make seasonal_freqs optional, only when we use MSTL or STL is this needed 
    mode: List[Literal['simple', 'mstl']] = 'simple'
):
    """
    Perform seasonal decomposition on a time series and compute additional features:
    - Trend strength
    - Seasonal strength
    - Residual variance

    Parameters:
    ----------
    df : pl.DataFrame
        A Polars DataFrame containing the time series data with the following columns:
        - `id_col`: Identifier for each time series (e.g., product, region, etc.)
        - `time_col`: Timestamp column indicating the time of each observation.
        - `target_col`: The column containing the values to be decomposed.

    id_col : str
        The name of the column that identifies each individual time series within the DataFrame.

    time_col : str
        The name of the column that contains the time or datetime information for each observation.

    target_col : str
        The name of the column containing the time series data to be decomposed.

    ts_freq : int
        The frequency of the time series. For example, `24` could represent daily seasonality in an hourly time series.

    seasonal_freqs : Optional[List[int]], optional
        A list of seasonal frequencies to use for the MSTL mode. This is required only if `mode='mstl'`.
        Defaults to None.

    mode : list of {'simple', 'mstl'}, optional
        The decomposition mode. Defaults to 'simple'.

    Returns:
    -------
    pl.DataFrame
        A DataFrame containing the following features for each unique `id_col`:
        - `trend_strength`: A measure of how strong the trend component is in the time series (value between 0 and 1).
        - `seasonal_strength`: A measure of how strong the seasonal component is in the time series (value between 0 and 1).
        - `resid_var`: The ratio of the standard deviation of the residuals to the mean of the target variable (a measure of residual variance).
    """

    # Check if necessary columns exist in the dataframe
    required_columns = [id_col, target_col, time_col]

    assert set(required_columns).issubset(df.columns), KeyError(f"Columns {set(required_columns).difference(df.columns)} are missing from the DataFrame.")

    # Validate ts_freq: ensure it's a positive integer
    if not isinstance(ts_freq, int) or ts_freq <= 0:
        raise ValueError(f"Invalid ts_freq '{ts_freq}'. It must be a positive integer.")
    
        # Ensure the dataframe is not empty
    if df.shape[0] == 0:
        raise ValueError("The DataFrame is empty. Cannot perform decomposition on an empty DataFrame.")

    assert mode in ['mstl','simple'], ValueError('Must Pick a mode "mstl" or "simple" to specify type of decomposition...')

    if seasonal_freqs is None and mode == 'mstl':
        raise ValueError('Must Specify atleast one seasonal freq in MSTL mode...')
    
    if seasonal_freqs is not None: 
        assert all(isinstance(freq,int) for freq in seasonal_freqs),'All Seasonal Frequencies must be integers'

    # Compute Trend Strength as MAX( 0, 1 - Var(R(t)) / Var(T(t) + R(t))) 
    trend_strength_expr = (
        pl.col('resid').var()
        .truediv(pl.col('trend').add(pl.col('resid')).var())
        .sub(1).abs().clip(lower_bound=0)
        .over(id_col).alias('trend_strength')
    )

    # Compute Resid Var as Std(R(t)) / Mean(y)
    resid_var_expr = (
        pl.col('resid').std().truediv(pl.col(target_col).mean()).over(id_col).alias('resid_var')
    )

    # Compute Seasonal Strength MAX(0, 1 - Var(R(t)) / Var(S(t) + R(t))) 
    seasonal_strength_expr = (
        pl.col('resid').var()
        .truediv(pl.col(f'seasonal').add(pl.col('resid')).var())
        .sub(1).abs().clip(lower_bound=0)
        .over(id_col).alias('seasonal_strength')
    )

    # Simple Decomposition Mode
    if mode == 'simple': 
        seas_decomp = seasonal_decomposition(df=df, id_col=id_col, time_col=time_col, target_col=target_col, freq=ts_freq)
        
        feats = (
            seas_decomp
            .with_columns(trend_strength_expr, seasonal_strength_expr, resid_var_expr)
            .select(id_col, 'trend_strength', 'seasonal_strength', 'resid_var')
            .unique(id_col)
        )

    # MSTL Mode (Only populate seasonal_freqs if mstl is active)
    elif mode == 'mstl':
        if seasonal_freqs is None:
            raise ValueError("For 'mstl' mode, 'seasonal_freqs' must be provided.")

        #map integer frequency to known datetime offset in polars: 

        freq_mapper = {12:'1mo',
                        52:'1w',
                        4:'1q',
                        24:'1h'
                        }

        assert ts_freq in freq_mapper.keys(), ValueError(f'Frequency must be one of {freq_mapper.keys()} for MSTL decompose..')

        #force columns to be renamed 
        col_remapper = {id_col:'unique_id',
        time_col:'ds',
        target_col:'y'}

        df = df.rename(col_remapper)

        # fit MSTL MODEL 
        model = MSTL(season_length=seasonal_freqs)

        # MSTL decomposition (assuming mstl_decomposition is defined)
        mstl_decomp = mstl_decomposition(df=df,model = model, freq=freq_mapper[ts_freq], h=1)[0]

        resid_expr = pl.col('y').sub(pl.sum_horizontal('trend', cs.contains('seasonal'))).alias('resid')

        # Compute Seasonal Strength for each seasonal component, for MSTL we compute it separately for each component.. 
        seasonal_strength_expr = [
            pl.col('resid').var()
            .truediv(pl.sum_horizontal('resid', seas_component).var())
            .sub(1).abs().clip(lower_bound=0)
            .over(id_col).alias(f'{seas_component}_strength')
            for seas_component in mstl_decomp.select(cs.contains('seasonal')).columns
        ]

        feats = (
            mstl_decomp
            .with_columns(resid_expr)
            .with_columns(seasonal_strength_expr)
            .with_columns(trend_strength_expr, resid_var_expr)
            .select('unique_id', cs.ends_with('_strength'), 'resid_var')
            .unique('unique_id')
        )

    return feats