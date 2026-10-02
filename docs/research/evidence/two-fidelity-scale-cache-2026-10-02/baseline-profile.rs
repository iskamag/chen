#[allow(dead_code)]
#[path="baseline-two_fidelity.rs"]
mod two_fidelity;
// Reproducible synthetic profiling; no chess strength or rating inference.
use two_fidelity::{
    Config, FastOracle, FiniteTree, OracleContract, SlowOracle, StopReason, TwoFidelity,
};
use std::time::Instant;
#[derive(Clone)]
struct Tree {
    children: Vec<Vec<usize>>,
}
impl FiniteTree for Tree {
    fn node_count(&self) -> usize {
        self.children.len()
    }
    fn children(&self, v: usize) -> &[usize] {
        &self.children[v]
    }
}
struct Fast(Vec<f64>);
impl FastOracle for Fast {
    fn evaluate(&mut self, v: usize) -> Result<f64, String> {
        Ok(self.0[v])
    }
}
struct Slow {
    means: Vec<f64>,
    noise: u64,
    sigma: f64,
}
impl SlowOracle for Slow {
    fn sample(&mut self, v: usize) -> Result<f64, String> {
        Ok(self.means[v]
            + if random(&mut self.noise) < 0.5 {
                -self.sigma
            } else {
                self.sigma
            })
    }
}
fn random(s: &mut u64) -> f64 {
    *s = s.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
}
fn build(d: usize, b: usize, seed: u64) -> (Tree, Vec<f64>, Vec<f64>) {
    let mut tree = Tree {
        children: vec![vec![]],
    };
    let mut depth = vec![0];
    let mut i = 0;
    while i < tree.node_count() {
        if depth[i] < d {
            for _ in 0..b {
                let u = tree.node_count();
                tree.children.push(vec![]);
                depth.push(depth[i] + 1);
                tree.children[i].push(u);
            }
        }
        i += 1;
    }
    let mut rng = seed;
    let mut values = vec![0.0; tree.node_count()];
    for v in (0..tree.node_count()).rev() {
        values[v] = if tree.children(v).is_empty() {
            random(&mut rng)
        } else {
            tree.children(v).iter().map(|&u| values[u]).fold(
                if depth[v] % 2 == 0 {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                },
                if depth[v] % 2 == 0 {
                    f64::max
                } else {
                    f64::min
                },
            )
        };
    }
    let fast = values
        .iter()
        .enumerate()
        .map(|(v, &x)| x + (2.0 * random(&mut rng) - 1.0) * 0.45 * (d - depth[v]) as f64 / d as f64)
        .collect();
    (tree, values, fast)
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let depth = args
        .get(1)
        .map_or(Ok(5), |s| s.parse::<usize>())
        .map_err(|e| e.to_string())?;
    let branch = args
        .get(2)
        .map_or(Ok(4), |s| s.parse::<usize>())
        .map_err(|e| e.to_string())?;
    let repeats = args
        .get(3)
        .map_or(Ok(5), |s| s.parse::<usize>())
        .map_err(|e| e.to_string())?;
    if depth == 0 || depth > 10 || branch < 2 || repeats == 0 || repeats > 1000 {
        return Err("usage: depth 1..10, branch >=2, repeats 1..1000".into());
    }
    let count = (0..=depth)
        .try_fold(0usize, |acc, h| {
            branch
                .checked_pow(h as u32)
                .and_then(|x| acc.checked_add(x))
        })
        .ok_or("tree size overflow")?;
    if count > 500_000 {
        return Err("profile tree exceeds 500000 nodes".into());
    }
    println!(
        "depth,branch,seed,nodes,stopped,correct,regret,fast_queries,slow_queries,cost,expansions,refresh_visits,certificate_visits,scale_evaluations,refresh_topology_scans,certificate_topology_scans,scale_storage_scans,wall_ms,fast_ms,slow_ms,refresh_ms,scale_ms,baseline_fast_queries,baseline_ms"
    );
    for seed in 0..repeats as u64 {
        let (tree, values, fast) = build(depth, branch, 20261002 + seed);
        let optimal = tree
            .children(0)
            .iter()
            .map(|&u| values[u])
            .fold(f64::NEG_INFINITY, f64::max);
        // Exhaustive minimax from exact fast leaf values. Internal heuristic
        // values are deliberately ignored. Cost includes every exposed node.
        let baseline_start = Instant::now();
        let mut baseline = fast.clone();
        let mut levels = vec![0; tree.node_count()];
        for v in 0..tree.node_count() {
            for &u in tree.children(v) {
                levels[u] = levels[v] + 1;
            }
        }
        for v in (0..tree.node_count()).rev() {
            if !tree.children(v).is_empty() {
                baseline[v] = tree.children(v).iter().map(|&u| baseline[u]).fold(
                    if levels[v] % 2 == 0 {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    },
                    if levels[v] % 2 == 0 {
                        f64::max
                    } else {
                        f64::min
                    },
                );
            }
        }
        let baseline_ms = baseline_start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(baseline[0], optimal);
        let n = tree.node_count();
        let mut c=TwoFidelity::new(tree,Fast(fast),Slow{means:values.clone(),noise:seed+73,sigma:0.01},
            Config { trace: args.get(4).is_some_and(|s| s == "trace"), ..Config::default() },OracleContract{envelope:(0..=depth).map(|h|0.45*h as f64/depth as f64).collect(),sigma:0.01,
                slow_bias_bound:0.0,provenance:"synthetic exact minimax plus seeded symmetric bounded noise; envelope verified by construction".into(),justified:true})?;
        let r = c.run()?;
        for (i, event) in r.trace.iter().enumerate() {
            eprintln!(
                "{seed},{i},{},{:?},{},{},{}",
                event.node, event.side, event.scale, event.expansion, event.cost
            );
        }
        let p = &r.profile;
        let regret = r
            .recommendation
            .map(|u| optimal - values[u])
            .unwrap_or(f64::NAN);
        println!(
            "{depth},{branch},{seed},{n},{},{},{regret},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{baseline_ms:.6}",
            r.stop == StopReason::Separated,
            regret == 0.0,
            p.fast_queries,
            p.slow_queries,
            r.total_cost,
            p.expansions,
            p.refresh_node_visits,
            p.certificate_node_visits,
            p.scale_evaluations,
            p.refresh_topology_scans,
            p.certificate_topology_scans,
            p.scale_storage_scans,
            p.total_time.as_secs_f64() * 1000.0,
            p.fast_time.as_secs_f64() * 1000.0,
            p.slow_time.as_secs_f64() * 1000.0,
            p.refresh_time.as_secs_f64() * 1000.0,
            p.scale_time.as_secs_f64() * 1000.0,
            n - 1
        );
    }
    Ok(())
}
