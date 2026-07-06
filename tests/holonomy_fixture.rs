//! Frozen seed-53 dataset fixture for the holonomy benchmark line.
//!
//! The 2026-07-01/02 Nagare holonomy results (entropy-pool local learner,
//! gate ablations, fitted projection gate) are all anchored on `make_dataset`
//! with seed 53 and the task-index seed offsets used by the compare example.
//! This fixture freezes those six datasets (3 tasks x train/test) as FNV-1a-64
//! content hashes over the exact f32 bit patterns, plus an 8-value hex preview.
//!
//! If `rand`'s `StdRng` stream or the generators ever drift, this test turns
//! the silent benchmark invalidation into a loud failure.
//!
//! The fixture records the platform it was frozen on (`# platform: arch-os`).
//! The exact float-bit hash is only asserted on that platform, because the
//! moons/spiral generators call `cos`/`sin`/`atan2`, whose last bit differs
//! across platform libm implementations (MSVC / glibc / Apple). On other
//! platforms the labels and structure are still checked strictly; only the
//! libm-dependent float hash is skipped.
//!
//! Regenerate deliberately with:
//! `cargo test --test holonomy_fixture -- --ignored`

use std::fmt::Write as _;
use std::path::PathBuf;

use holonomy_learn::{make_dataset, Dataset, Task};

const FIXTURE_SEED: u64 = 53;
const N_TRAIN: usize = 192;
const N_TEST: usize = 96;
const N_POINTS: usize = 32;
const PREVIEW_LEN: usize = 8;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("moons_spiral_xor_seed53.txt")
}

/// FNV-1a 64-bit over a byte stream.
fn fnv1a64(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn hash_x(data: &Dataset) -> u64 {
    fnv1a64(data.x.iter().flat_map(|v| v.to_bits().to_le_bytes()))
}

fn hash_y(data: &Dataset) -> u64 {
    fnv1a64(data.y.iter().flat_map(|&v| (v as u64).to_le_bytes()))
}

fn preview(data: &Dataset) -> String {
    data.x
        .iter()
        .take(PREVIEW_LEN)
        .map(|v| format!("{:08x}", v.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

/// The six benchmark datasets with the exact seed offsets of the compare
/// example: train `seed + idx * 100`, test `seed + idx * 100 + 1`.
fn fixture_cases() -> Vec<(Task, &'static str, usize, u64)> {
    [Task::Moons, Task::Spiral, Task::Xor]
        .iter()
        .enumerate()
        .flat_map(|(idx, &task)| {
            let base = FIXTURE_SEED + idx as u64 * 100;
            [
                (task, "train", N_TRAIN, base),
                (task, "test", N_TEST, base + 1),
            ]
        })
        .collect()
}

fn render_fixture() -> String {
    let mut out = String::new();
    out.push_str("# Frozen seed-53 moons/spiral/xor datasets (Nagare holonomy benchmark line)\n");
    out.push_str(&format!("# platform: {}\n", platform()));
    out.push_str("# task split samples points seed x_fnv1a64 y_fnv1a64 x_preview_hex\n");
    for (task, split, samples, seed) in fixture_cases() {
        let data = make_dataset(task, samples, N_POINTS, seed);
        writeln!(
            out,
            "{} {} {} {} {} {:016x} {:016x} {}",
            task.as_str(),
            split,
            samples,
            N_POINTS,
            seed,
            hash_x(&data),
            hash_y(&data),
            preview(&data)
        )
        .expect("string write cannot fail");
    }
    out
}

/// Target key the fixture floats are pinned to (`arch-os`, e.g.
/// `x86_64-windows`). Exact f32 bit patterns of the moons/spiral generators
/// depend on the platform libm's `cos`/`sin`/`atan2` (they differ by up to a
/// ULP between MSVC, glibc, and Apple libm), so the float-hash check is only
/// meaningful on the platform the fixture was frozen on.
fn platform() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

/// The platform the frozen fixture was generated on, read from its
/// `# platform:` header.
fn frozen_platform(frozen: &str) -> Option<String> {
    frozen
        .lines()
        .find_map(|l| l.strip_prefix("# platform:").map(|s| s.trim().to_string()))
}

/// Platform-independent fields of a fixture row: task, split, samples, points,
/// seed, and the *label* hash. These are integers or structural, so they hold
/// bit-for-bit across platforms — only the float-data hash (col 5) and the
/// float preview (col 7) carry libm-dependent bits.
fn platform_independent(line: &str) -> Vec<&str> {
    let f: Vec<&str> = line.split_whitespace().collect();
    assert!(f.len() >= 7, "malformed fixture row: {line}");
    vec![f[0], f[1], f[2], f[3], f[4], f[6]]
}

#[test]
fn seed53_datasets_match_frozen_fixture() {
    let path = fixture_path();
    let frozen = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()));
    let current = render_fixture();
    let frozen_lines: Vec<&str> = frozen.lines().filter(|l| !l.starts_with('#')).collect();
    let current_lines: Vec<&str> = current.lines().filter(|l| !l.starts_with('#')).collect();
    assert_eq!(
        frozen_lines.len(),
        current_lines.len(),
        "fixture row count changed"
    );

    let frozen_os = frozen_platform(&frozen).unwrap_or_else(|| {
        panic!("fixture is missing a `# platform:` header; regenerate with `-- --ignored`")
    });
    let same_platform = frozen_os == platform();

    for (frozen_line, current_line) in frozen_lines.iter().zip(current_lines.iter()) {
        if same_platform {
            // Strict, exact-bit check on the freeze platform.
            assert_eq!(
                frozen_line, current_line,
                "seed-53 dataset drifted from the frozen fixture; if the change is \
                 deliberate, regenerate with `-- --ignored` and re-anchor the benchmarks"
            );
        } else {
            // Cross-platform: labels + structure must still match bit-for-bit;
            // the libm-dependent float hash is not comparable across platforms.
            assert_eq!(
                platform_independent(frozen_line),
                platform_independent(current_line),
                "seed-53 labels/structure drifted (platform-independent fields); \
                 fixture frozen on '{frozen_os}', running on '{}'",
                platform()
            );
        }
    }

    if !same_platform {
        eprintln!(
            "note: exact float-bit check skipped — fixture frozen on '{frozen_os}', running on \
             '{}'. Transcendental libm (cos/sin/atan2) differs by ULPs across platforms; labels \
             and structure were verified, and downstream metrics are platform-robust (see the \
             order-shuffle ablation, which reproduces to 6 decimals cross-platform).",
            platform()
        );
    }
}

#[test]
#[ignore = "writes the fixture; run deliberately after an intentional generator change"]
fn regenerate_seed53_fixture() {
    let path = fixture_path();
    std::fs::create_dir_all(path.parent().expect("fixture path has a parent"))
        .expect("create fixtures dir");
    std::fs::write(&path, render_fixture()).expect("write fixture");
}
