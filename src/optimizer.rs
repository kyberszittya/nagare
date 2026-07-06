//! Optimizer module (currently a thin re-export; will host SGD,
//! AdamW, LAMB as Nagare grows).
pub use crate::ops::adam::{adam_step, AdamState};
