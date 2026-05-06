/// KASBA: K-means with Accelerated and Stochastic Barycenter Averaging.
///
/// Ported from <https://github.com/alirezakarimi82/KASBA-Rust>.
/// Coexists with existing `src/msm.rs` (different consumer — Polars DataFrame pairwise API).
pub mod alignment;
pub mod assign;
pub mod average;
pub mod clustering;
pub mod empty_cluster;
pub mod init;
pub mod msm;
pub mod pairwise;

#[cfg(test)]
mod tests;
