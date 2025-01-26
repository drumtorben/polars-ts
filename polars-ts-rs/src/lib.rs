use pyo3_polars::PolarsAllocator;

mod mann_kendall;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();