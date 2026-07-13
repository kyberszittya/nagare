//! Learned 2-D convolution — the **S-cell** of the Nagare Neocognitron. A
//! multi-channel, zero-padded, stride-1 convolution with a hand-derived,
//! FD-verified backward (no autograd). This is the feature-extraction primitive
//! the pose P1 report identified as missing (joint-discriminative spatial
//! features).
//!
//! Layout: channel-first flat. Input `x ∈ (C_i, H, W)`, kernel
//! `K ∈ (C_o, C_i, kh, kw)`, bias `b ∈ (C_o)`, zero-pad `p`, stride 1.
//! `H' = H + 2p - kh + 1`, `W' = W + 2p - kw + 1`:
//! ```text
//! y[o,h',w'] = b[o] + Σ_{i,a,c} K[o,i,a,c] · x[i, h'+a-p, w'+c-p]   (0 outside the image)
//! ```
//!
//! # Backward (FD-verified)
//! ```text
//! b_bar[o]       = Σ_{h',w'} y_bar[o,h',w']
//! K_bar[o,i,a,c] = Σ_{h',w'} y_bar[o,h',w'] · x[i, h'+a-p, w'+c-p]
//! x_bar[i,h,w]   = Σ_{o,a,c} y_bar[o, h-a+p, w-c+p] · K[o,i,a,c]      (transposed correlation)
//! ```
//! All three are accumulated in one bounds-checked pass.
//!
//! **No novelty claimed** (conv2d is universal; Neocognitron = Fukushima 1980).
//! The Nagare-specific part is the closed-form no-autograd op, and its role as
//! the S-cell of a rotation-equivariant Neocognitron whose C-cells reuse the
//! crate's rotor ops (dihedral group, quaternion rotors, `rotor_spike`).

use rand::{Rng, SeedableRng};

/// Convolution parameters (also used as the gradient buffer, like `LinearLayer`).
#[derive(Clone, Debug)]
pub struct ConvLayer {
    /// Kernel weights `(c_out, c_in, kh, kw)` flat.
    pub w: Vec<f32>,
    /// Biases `(c_out,)`.
    pub b: Vec<f32>,
    pub c_in: usize,
    pub c_out: usize,
    pub kh: usize,
    pub kw: usize,
}

impl ConvLayer {
    /// New layer with Glorot-uniform init.
    pub fn new(c_in: usize, c_out: usize, kh: usize, kw: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let fan = (c_in + c_out) * kh * kw;
        let scale = (6.0 / fan as f32).sqrt();
        let w = (0..c_out * c_in * kh * kw)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
            .collect();
        ConvLayer {
            w,
            b: vec![0.0; c_out],
            c_in,
            c_out,
            kh,
            kw,
        }
    }
    /// Zero gradient buffer of the same shape.
    pub fn zero_grad(&self) -> ConvLayer {
        ConvLayer {
            w: vec![0.0; self.w.len()],
            b: vec![0.0; self.b.len()],
            c_in: self.c_in,
            c_out: self.c_out,
            kh: self.kh,
            kw: self.kw,
        }
    }
}

/// Input spatial shape + zero-padding for a conv call.
#[derive(Clone, Copy, Debug)]
pub struct ConvShape {
    pub c_in: usize,
    pub h: usize,
    pub w: usize,
    pub pad: usize,
}

impl ConvShape {
    /// Output `(H', W')` for a kernel of size `(kh, kw)`.
    pub fn out_hw(&self, kh: usize, kw: usize) -> (usize, usize) {
        (
            self.h + 2 * self.pad - kh + 1,
            self.w + 2 * self.pad - kw + 1,
        )
    }
}

/// Conv2d forward. `x` is `(c_in, h, w)` flat; returns `(y (c_out,H',W'), H', W')`.
///
/// # Panics
/// If `x.len() != c_in*h*w` or the layer's `c_in` disagrees with the shape.
pub fn conv2d_forward(l: &ConvLayer, x: &[f32], s: ConvShape) -> (Vec<f32>, usize, usize) {
    assert_eq!(x.len(), s.c_in * s.h * s.w);
    assert_eq!(l.c_in, s.c_in);
    let (oh, ow) = s.out_hw(l.kh, l.kw);
    let (h, w, pad) = (s.h as isize, s.w as isize, s.pad as isize);
    let mut y = vec![0.0f32; l.c_out * oh * ow];
    for co in 0..l.c_out {
        for yh in 0..oh {
            for yw in 0..ow {
                let mut acc = l.b[co];
                for ci in 0..l.c_in {
                    for a in 0..l.kh {
                        let ih = yh as isize + a as isize - pad;
                        if ih < 0 || ih >= h {
                            continue;
                        }
                        for c in 0..l.kw {
                            let iw = yw as isize + c as isize - pad;
                            if iw < 0 || iw >= w {
                                continue;
                            }
                            let kv = l.w[((co * l.c_in + ci) * l.kh + a) * l.kw + c];
                            acc += kv * x[(ci * s.h + ih as usize) * s.w + iw as usize];
                        }
                    }
                }
                y[(co * oh + yh) * ow + yw] = acc;
            }
        }
    }
    (y, oh, ow)
}

/// Conv2d backward. Returns `(grad_input (c_in,h,w), grad_layer)`.
///
/// # Panics
/// If `grad_y.len() != c_out*H'*W'`.
pub fn conv2d_backward(
    l: &ConvLayer,
    x: &[f32],
    s: ConvShape,
    grad_y: &[f32],
) -> (Vec<f32>, ConvLayer) {
    let (oh, ow) = s.out_hw(l.kh, l.kw);
    assert_eq!(grad_y.len(), l.c_out * oh * ow);
    let (h, w, pad) = (s.h as isize, s.w as isize, s.pad as isize);
    let mut gx = vec![0.0f32; s.c_in * s.h * s.w];
    let mut gl = l.zero_grad();
    for co in 0..l.c_out {
        for yh in 0..oh {
            for yw in 0..ow {
                let gy = grad_y[(co * oh + yh) * ow + yw];
                gl.b[co] += gy;
                for ci in 0..l.c_in {
                    for a in 0..l.kh {
                        let ih = yh as isize + a as isize - pad;
                        if ih < 0 || ih >= h {
                            continue;
                        }
                        for c in 0..l.kw {
                            let iw = yw as isize + c as isize - pad;
                            if iw < 0 || iw >= w {
                                continue;
                            }
                            let xi = (ci * s.h + ih as usize) * s.w + iw as usize;
                            let ki = ((co * l.c_in + ci) * l.kh + a) * l.kw + c;
                            gl.w[ki] += gy * x[xi];
                            gx[xi] += gy * l.w[ki];
                        }
                    }
                }
            }
        }
    }
    (gx, gl)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(x: &mut [f32], seed: u64) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for v in x.iter_mut() {
            *v = rng.random::<f32>() * 2.0 - 1.0;
        }
    }

    #[test]
    fn grads_match_fd() {
        let l = ConvLayer::new(2, 3, 3, 3, 5);
        let s = ConvShape {
            c_in: 2,
            h: 5,
            w: 5,
            pad: 1,
        };
        let mut x = vec![0.0f32; s.c_in * s.h * s.w];
        seeded(&mut x, 9);
        let (y, oh, ow) = conv2d_forward(&l, &x, s);
        let mut go = vec![0.0f32; y.len()];
        seeded(&mut go, 13); // arbitrary upstream gradient; L = <go, y>
        let (gx, gl) = conv2d_backward(&l, &x, s, &go);
        let loss = |ll: &ConvLayer, xx: &[f32]| -> f32 {
            conv2d_forward(ll, xx, s)
                .0
                .iter()
                .zip(&go)
                .map(|(&a, &b)| a * b)
                .sum()
        };
        let eps = 1e-3;
        let chk = |ana: f32, num: f32, what: &str, i: usize| {
            assert!(
                (ana - num).abs() < 1e-3 + 2e-2 * num.abs(),
                "{what}[{i}] {ana} vs {num}"
            );
        };
        // grad_input
        for i in 0..x.len() {
            let (mut a, mut b) = (x.clone(), x.clone());
            a[i] += eps;
            b[i] -= eps;
            chk(gx[i], (loss(&l, &a) - loss(&l, &b)) / (2.0 * eps), "gx", i);
        }
        // grad_kernel
        for i in 0..l.w.len() {
            let (mut la, mut lb) = (l.clone(), l.clone());
            la.w[i] += eps;
            lb.w[i] -= eps;
            chk(
                gl.w[i],
                (loss(&la, &x) - loss(&lb, &x)) / (2.0 * eps),
                "gw",
                i,
            );
        }
        // grad_bias
        for i in 0..l.b.len() {
            let (mut la, mut lb) = (l.clone(), l.clone());
            la.b[i] += eps;
            lb.b[i] -= eps;
            chk(
                gl.b[i],
                (loss(&la, &x) - loss(&lb, &x)) / (2.0 * eps),
                "gb",
                i,
            );
        }
        let _ = (oh, ow);
    }

    #[test]
    fn identity_kernel_is_input() {
        let mut l = ConvLayer::new(1, 1, 1, 1, 0);
        l.w = vec![1.0];
        l.b = vec![0.0];
        let s = ConvShape {
            c_in: 1,
            h: 4,
            w: 4,
            pad: 0,
        };
        let mut x = vec![0.0f32; 16];
        seeded(&mut x, 3);
        let (y, oh, ow) = conv2d_forward(&l, &x, s);
        assert_eq!((oh, ow), (4, 4));
        for (a, b) in y.iter().zip(&x) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn shift_equivariance() {
        // A 3x3 filter, pad 1: shifting the input one column shifts the output.
        let l = ConvLayer::new(1, 1, 3, 3, 7);
        let s = ConvShape {
            c_in: 1,
            h: 6,
            w: 6,
            pad: 1,
        };
        let mut x = vec![0.0f32; 36];
        seeded(&mut x, 11);
        let (y0, _, ow) = conv2d_forward(&l, &x, s);
        // shift x right by 1 column (drop last col, zero first)
        let mut xs = vec![0.0f32; 36];
        for r in 0..6 {
            for c in 1..6 {
                xs[r * 6 + c] = x[r * 6 + c - 1];
            }
        }
        let (y1, _, _) = conv2d_forward(&l, &xs, s);
        // interior columns of y1 equal the shifted interior of y0.
        for r in 1..5 {
            for c in 2..5 {
                assert!(
                    (y1[r * ow + c] - y0[r * ow + c - 1]).abs() < 1e-5,
                    "shift r{r} c{c}"
                );
            }
        }
    }
}
