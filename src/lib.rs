use pyo3_polars::PolarsAllocator;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use dtw::compute_pairwise_dtw;
use ddtw::compute_pairwise_ddtw;
use wdtw::compute_pairwise_wdtw;
use msm::compute_pairwise_msm;
use dtw_multi::compute_pairwise_dtw_multi;
use msm_multi::compute_pairwise_msm_multi;
use erp::compute_pairwise_erp;
use lcss::compute_pairwise_lcss;
use twe::compute_pairwise_twe;
use sbd::compute_pairwise_sbd;
use frechet::compute_pairwise_frechet;
use edr::compute_pairwise_edr;

mod utils;
mod dtw;
mod dtw_multi;
mod msm;
mod msm_multi;
mod ddtw;
mod wdtw;
mod erp;
mod lcss;
mod twe;
mod sbd;
mod frechet;
mod edr;
mod mann_kendall;
mod sens_slope;
mod ets;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "polars_ts_rs")]
fn polars_ts_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_pairwise_dtw, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pairwise_msm, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pairwise_ddtw, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pairwise_wdtw, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pairwise_dtw_multi, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pairwise_msm_multi, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pairwise_erp, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pairwise_lcss, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pairwise_twe, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pairwise_sbd, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pairwise_frechet, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pairwise_edr, m)?)?;
    m.add_function(wrap_pyfunction!(ets::ets_ses, m)?)?;
    m.add_function(wrap_pyfunction!(ets::ets_holt, m)?)?;
    m.add_function(wrap_pyfunction!(ets::ets_holt_winters, m)?)?;
    Ok(())
}
