use pyo3_polars::PolarsAllocator;
// use pyo3_polars::*;
// use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use dtw::compute_pairwise_dtw;

mod dtw;
mod mann_kendall;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

// A Python module implemented in Rust.
// #[pymodule]
// fn polars_ts_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(dtw::my_cool_function, m)?)?;
//     Ok(())
// }


// #[pyfunction]
// fn my_cool_function(pydf: PyDataFrame, pydf2: PyDataFrame) -> PyResult<PyDataFrame> {
//     let df: DataFrame = pydf.into();
//     let df2: DataFrame = pydf2.into();
//     println!("{:?}", df);
//     println!("{:?}", df2);
//     // wrap the dataframe and it will be automatically converted to a python polars dataframe
//     Ok(PyDataFrame(df))
// }

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "polars_ts_rs")]
fn polars_ts_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    //m.add_function(wrap_pyfunction!(my_cool_function, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pairwise_dtw, m)?)?;
    
    Ok(())
}