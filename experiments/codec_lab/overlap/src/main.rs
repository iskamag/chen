//! Tiny exact visibility experiment. No chess engine or search is linked.
use serde_json::{Value, json};
use std::{collections::HashSet, hint::black_box, time::Instant};

type Line = [u8; 16];
type Relation = [u16; 16];

// This oracle walks outwards from each actual source. The band/control code
// instead sweeps destinations and carries the nearest occupied source.
fn oracle(x: &Line, n: usize) -> Relation {
    let mut out = [0; 16];
    for source in 0..n {
        if x[source] != 2 {
            continue;
        }
        for direction in [-1, 1] {
            let mut target = source as i32 + direction;
            while target >= 0 && target < n as i32 {
                out[source] |= 1 << target;
                if x[target as usize] != 0 {
                    break;
                }
                target += direction;
            }
        }
    }
    out
}

fn band(x: &Line, n: usize) -> Relation {
    let mut out = [0; 16];
    for reverse in [false, true] {
        let mut source = None;
        for k in 0..n {
            let target = if reverse { n - 1 - k } else { k };
            if let Some(s) = source {
                out[s] |= 1 << target;
            }
            if x[target] != 0 {
                source = if x[target] == 2 { Some(target) } else { None };
            }
        }
    }
    out
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Block {
    local: [u8; 4],
    prefix: u8,
    suffix: u8,
    left_exit: u8, // last occupied slider +1, or inactive
    right_exit: u8,
    empty: bool,
}

fn decode(mut code: usize, n: usize) -> Line {
    let mut x = [0; 16];
    for cell in x.iter_mut().take(n) {
        *cell = (code % 3) as u8;
        code /= 3;
    }
    x
}

fn code(x: &[u8]) -> usize {
    x.iter().rev().fold(0, |v, &cell| 3 * v + cell as usize)
}

fn make_block(x: &Line) -> Block {
    let relation = band(x, 4);
    let mut prefix = 0;
    for (i, &cell) in x.iter().enumerate().take(4) {
        prefix |= 1 << i;
        if cell != 0 {
            break;
        }
    }
    let mut suffix = 0;
    for i in (0..4).rev() {
        suffix |= 1 << i;
        if x[i] != 0 {
            break;
        }
    }
    let left_exit = (0..4)
        .rev()
        .find(|&i| x[i] != 0)
        .filter(|&i| x[i] == 2)
        .map_or(0, |i| i as u8 + 1);
    let right_exit = (0..4)
        .find(|&i| x[i] != 0)
        .filter(|&i| x[i] == 2)
        .map_or(0, |i| i as u8 + 1);
    Block {
        local: std::array::from_fn(|i| relation[i] as u8),
        prefix,
        suffix,
        left_exit,
        right_exit,
        empty: x[..4].iter().all(|&v| v == 0),
    }
}

fn compose(blocks: &[Block]) -> Relation {
    let mut out = [0; 16];
    for (b, block) in blocks.iter().enumerate() {
        for s in 0..4 {
            out[b * 4 + s] |= (block.local[s] as u16) << (b * 4);
        }
    }
    for reverse in [false, true] {
        let mut active: Option<usize> = None;
        for k in 0..blocks.len() {
            let b = if reverse { blocks.len() - 1 - k } else { k };
            let block = blocks[b];
            if let Some(s) = active {
                out[s] |= (if reverse { block.suffix } else { block.prefix } as u16) << (b * 4);
            }
            if !block.empty {
                let exit = if reverse {
                    block.right_exit
                } else {
                    block.left_exit
                };
                active = (exit != 0).then(|| b * 4 + exit as usize - 1);
            }
        }
    }
    out
}

fn boundary(x: &Line, n: usize, table: &[Block]) -> Relation {
    let mut blocks = [table[0]; 4];
    for (i, chunk) in x[..n].chunks_exact(4).enumerate() {
        blocks[i] = table[code(chunk)];
    }
    compose(&blocks[..n / 4])
}

fn local(x: &Line, n: usize, table: &[Block], stride: usize, mean: bool) -> Relation {
    if !mean {
        let mut out = [0; 16];
        for start in (0..=n - 4).step_by(stride) {
            let block = table[code(&x[start..start + 4])];
            for s in 0..4 {
                out[start + s] |= (block.local[s] as u16) << start;
            }
        }
        return out;
    }
    let mut votes = [[0u8; 16]; 16];
    let mut coverage = [0u8; 16];
    for start in (0..=n - 4).step_by(stride) {
        let b = table[code(&x[start..start + 4])];
        for t in start..start + 4 {
            coverage[t] += 1;
        }
        for s in 0..4 {
            for t in 0..4 {
                if b.local[s] & (1 << t) != 0 {
                    votes[start + s][start + t] += 1;
                }
            }
        }
    }
    let mut out = [0; 16];
    for s in 0..n {
        for t in 0..n {
            if if mean {
                votes[s][t] * 2 > coverage[t]
            } else {
                votes[s][t] != 0
            } {
                out[s] |= 1 << t;
            }
        }
    }
    out
}

// A toy lifting construction inspired by JPEG XR's integer shears. It is NOT
// the JPEG XR POT/PCT. Across hard boundaries use two invertible integer shears;
// then reversible pair S/Haar lifting. Every inverse uses the same floor shift.
fn lift(x: &Line, n: usize) -> [i16; 16] {
    let mut y = x.map(|v| match v {
        0 => 0,
        1 => -1,
        _ => 1,
    });
    for b in (4..n).step_by(4) {
        y[b] -= y[b - 1];
        y[b - 1] += y[b] >> 1;
    }
    for i in (0..n).step_by(2) {
        let d = y[i + 1] - y[i];
        y[i] += d >> 1;
        y[i + 1] = d;
    }
    y
}

fn unlift(mut y: [i16; 16], n: usize, dc_only: bool) -> Line {
    if dc_only {
        for i in (1..n).step_by(2) {
            y[i] = 0;
        }
    }
    for i in (0..n).step_by(2) {
        let a = y[i] - (y[i + 1] >> 1);
        y[i + 1] += a;
        y[i] = a;
    }
    for b in (4..n).step_by(4).rev() {
        y[b - 1] -= y[b] >> 1;
        y[b] += y[b - 1];
    }
    y.map(|v| {
        if v == 0 {
            0
        } else if v < 0 {
            1
        } else {
            2
        }
    })
}

fn variant(name: &str, x: &Line, n: usize, table: &[Block], full8: &[Relation]) -> Relation {
    match name {
        "direct_source_ray_oracle" => oracle(x, n),
        "directional_band_operator" => band(x, n),
        "full8_lut" => full8[code(&x[..8])],
        "hard4_local" => local(x, n, table, 4, false),
        "stagger4_stride2_or" => local(x, n, table, 2, false),
        "stagger4_stride2_mean_majority" => local(x, n, table, 2, true),
        "boundary4_lut" => boundary(x, n, table),
        "lapped_lifting_inverse_boundary4" => boundary(&unlift(lift(x, n), n, false), n, table),
        "lifting_dc_only_boundary4" => boundary(&unlift(lift(x, n), n, true), n, table),
        _ => unreachable!(),
    }
}

#[derive(Default)]
struct Errors {
    roots: usize,
    whole: usize,
    bits: usize,
    fp: usize,
    fn_: usize,
    edge: usize,
    long: usize,
    example: Option<Value>,
}

impl Errors {
    fn add(&mut self, x: &Line, n: usize, expected: Relation, actual: Relation) {
        self.roots += 1;
        if expected == actual {
            return;
        }
        self.whole += 1;
        for s in 0..n {
            self.bits += (expected[s] ^ actual[s]).count_ones() as usize;
            self.fp += (actual[s] & !expected[s]).count_ones() as usize;
            self.fn_ += (expected[s] & !actual[s]).count_ones() as usize;
            for t in 0..n {
                if ((expected[s] ^ actual[s]) >> t) & 1 == 0 {
                    continue;
                }
                if s.abs_diff(t) == 1 && s / 4 != t / 4 {
                    self.edge += 1;
                }
                if s.abs_diff(t) >= 4 {
                    self.long += 1;
                }
            }
        }
        if self.example.is_none() {
            self.example =
                Some(json!({"line": &x[..n], "expected": &expected[..n], "actual": &actual[..n]}));
        }
    }
    fn json(&self) -> Value {
        json!({"roots":self.roots,"whole_relation_errors":self.whole,"wrong_relation_bits":self.bits,
            "false_positive_bits":self.fp,"false_negative_bits":self.fn_,"distance1_cross_hard_edge_errors":self.edge,
            "distance_at_least4_errors":self.long,"first_counterexample":self.example})
    }
}

fn class_oracles(x: &Line) -> Vec<Relation> {
    let mut out = Vec::with_capacity(48);
    for at in 0..16 {
        for value in 0..3 {
            let mut y = *x;
            y[at] = value;
            out.push(oracle(&y, 16));
        }
    }
    out
}

fn class_boundary(x: &Line, table: &[Block], shared: bool) -> Vec<Relation> {
    let codes: [usize; 4] = std::array::from_fn(|i| code(&x[i * 4..i * 4 + 4]));
    let original: [Block; 4] = codes.map(|c| table[c]);
    let mut out = Vec::with_capacity(48);
    for at in 0..16 {
        for value in 0..3 {
            if shared {
                let mut blocks = original;
                let new_code = codes[at / 4] as i32
                    + (value as i32 - x[at] as i32) * 3i32.pow((at % 4) as u32);
                blocks[at / 4] = table[new_code as usize];
                out.push(compose(&blocks));
            } else {
                let mut y = *x;
                y[at] = value;
                out.push(boundary(&y, 16, table));
            }
        }
    }
    out
}

fn timed<F: FnMut()>(mut f: F, iterations: usize) -> Value {
    let mut samples = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..iterations {
            f();
        }
        samples.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    json!({"median_ns":samples[3],"min_ns":samples[0],"max_ns":samples[6],"repetitions":7})
}

fn main() {
    let table: Vec<_> = (0..81).map(|i| make_block(&decode(i, 4))).collect();
    let full8: Vec<_> = (0..6561).map(|i| band(&decode(i, 8), 8)).collect();
    let all8: Vec<_> = (0..6561).map(|i| decode(i, 8)).collect();
    let mut random = Vec::new();
    let mut seed = 12587763u64;
    for _ in 0..20000 {
        let mut x = [0; 16];
        for cell in &mut x {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *cell = (seed % 3) as u8;
        }
        random.push(x);
    }
    let mut constructed = HashSet::new();
    for s in 0..16 {
        let mut x = [0; 16];
        x[s] = 2;
        constructed.insert(x);
        for t in 0..16 {
            if s == t {
                continue;
            }
            for value in [1, 2] {
                let mut y = x;
                y[t] = value;
                constructed.insert(y);
            }
        }
    }
    let mut constructed: Vec<_> = constructed.into_iter().collect();
    constructed.sort();
    let names = [
        "direct_source_ray_oracle",
        "directional_band_operator",
        "full8_lut",
        "hard4_local",
        "stagger4_stride2_or",
        "stagger4_stride2_mean_majority",
        "boundary4_lut",
        "lapped_lifting_inverse_boundary4",
        "lifting_dc_only_boundary4",
    ];
    let mut metrics = serde_json::Map::new();
    let mut roundtrip = 0;
    for (family, inputs, n) in [
        ("exhaustive8", &all8, 8),
        ("random16", &random, 16),
        ("constructed16", &constructed, 16),
    ] {
        let mut family_metrics = serde_json::Map::new();
        for name in names {
            if name == "full8_lut" && n != 8 {
                continue;
            }
            let mut errors = Errors::default();
            for x in inputs {
                errors.add(x, n, oracle(x, n), variant(name, x, n, &table, &full8));
            }
            family_metrics.insert(name.into(), errors.json());
        }
        for x in inputs {
            roundtrip += usize::from(unlift(lift(x, n), n, false) != *x);
        }
        metrics.insert(family.into(), Value::Object(family_metrics));
    }
    let mut class_metrics = serde_json::Map::new();
    let class_inputs: Vec<_> = constructed
        .iter()
        .chain(random[..256].iter())
        .copied()
        .collect();
    for (name, shared) in [
        ("boundary4_rebuild_each_query", false),
        ("boundary4_shared_descriptor_edits", true),
    ] {
        let mut query_errors = 0;
        let mut class_errors = 0;
        for x in &class_inputs {
            let expected = class_oracles(x);
            let actual = class_boundary(x, &table, shared);
            let bad = expected.iter().zip(&actual).filter(|(a, b)| a != b).count();
            query_errors += bad;
            class_errors += usize::from(bad != 0);
        }
        class_metrics.insert(name.into(),json!({"roots":class_inputs.len(),"queries":class_inputs.len()*48,
            "complete_class_errors":class_errors,"whole_relation_query_errors":query_errors,
            "local_code_extractions_per_class":if shared {4} else {192},"local_table_lookups_per_class":if shared {52} else {192},
            "boundary_compositions_per_class":48}));
    }
    let mut timing = serde_json::Map::new();
    for n in [8, 16] {
        let inputs = if n == 8 { &all8[..] } else { &random[..4096] };
        let mut rows = serde_json::Map::new();
        for name in names {
            if name == "full8_lut" && n != 8 {
                continue;
            }
            let t = timed(
                || {
                    for x in inputs {
                        black_box(variant(black_box(name), black_box(x), n, &table, &full8));
                    }
                },
                20,
            );
            let mut t = t.as_object().unwrap().clone();
            for key in ["median_ns", "min_ns", "max_ns"] {
                t.insert(
                    key.into(),
                    json!(t[key].as_f64().unwrap() / inputs.len() as f64),
                );
            }
            rows.insert(name.into(), Value::Object(t));
        }
        timing.insert(format!("n{n}_complete_relation"), Value::Object(rows));
    }
    let mut class_timing = serde_json::Map::new();
    for name in [
        "independent_source_oracles",
        "boundary4_rebuild_each_query",
        "boundary4_shared_descriptor_edits",
    ] {
        let t = timed(
            || {
                for x in &random[..256] {
                    black_box(match name {
                        "independent_source_oracles" => class_oracles(black_box(x)),
                        "boundary4_rebuild_each_query" => {
                            class_boundary(black_box(x), &table, false)
                        }
                        _ => class_boundary(black_box(x), &table, true),
                    });
                }
            },
            20,
        );
        let mut t = t.as_object().unwrap().clone();
        for key in ["median_ns", "min_ns", "max_ns"] {
            t.insert(key.into(), json!(t[key].as_f64().unwrap() / 256.0));
        }
        class_timing.insert(name.into(), Value::Object(t));
    }
    let distinct = table.iter().collect::<HashSet<_>>().len();
    println!("{}",serde_json::to_string_pretty(&json!({"correctness":metrics,"action_classes":class_metrics,
        "transform_roundtrip_errors":roundtrip,"storage":{"local4_entries":81,"local4_descriptor_bytes":std::mem::size_of::<Block>(),
        "local4_table_bytes":81*std::mem::size_of::<Block>(),"distinct_boundary_operator_descriptors":distinct,
        "full8_entries":6561,"full8_physical_bytes":full8.len()*std::mem::size_of::<Relation>(),
        "full8_logical_packed_bytes":6561*8,"full16_entries_not_allocated":3u64.pow(16),
        "full16_logical_packed_bytes_not_allocated":3u64.pow(16)*32,
        "directional_boundary_states_inactive_plus_sourceidentity":17,
        "note":"Block descriptor identity is exact full local operator, not an assertion of minimal encoding"},
        "timing":timing,"class_timing_ns_per48queries":class_timing})).unwrap());
}
