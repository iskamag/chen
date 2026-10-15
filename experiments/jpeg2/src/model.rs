//! Direct learned heterogeneous-channel factors. No chess transitions or search.
use crate::features::DIM;
use serde_json::Value;
use std::{fs, path::Path};
#[derive(Clone)]
struct Linear {
    rows: usize,
    cols: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}
#[derive(Clone)]
pub struct Group {
    pub name: String,
    pub start: usize,
    pub end: usize,
}
pub struct Model {
    pub name: String,
    pub groups: Vec<Group>,
    width: usize,
    mean: Vec<f32>,
    inverse_std: Vec<f32>,
    clip: f32,
    projections: Vec<Linear>,
    base: Vec<f32>,
    base_bias: f32,
    pairs: Vec<(usize, usize)>,
    residual: Option<(Linear, Vec<f32>, f32)>,
}
#[derive(Clone, Default)]
pub struct Controls {
    pub base_only: bool,
    pub pair_products_off: bool,
    pub omit_channel: Option<String>,
}
#[derive(Clone)]
pub struct Scores {
    pub logits: Vec<f32>,
    pub base_logits: Vec<f32>,
    pub base_best: Option<usize>,
    pub full_best: Option<usize>,
    pub interaction_terms: usize,
}
fn number(v: &Value) -> Result<f32, String> {
    let f = v.as_f64().ok_or("expected numeric model weight")? as f32;
    if f.is_finite() {
        Ok(f)
    } else {
        Err("non-finite model weight".into())
    }
}
fn vector(v: &Value, len: usize) -> Result<Vec<f32>, String> {
    let a = v.as_array().ok_or("expected weight array")?;
    if a.len() != len {
        return Err(format!("weight length {}, expected {len}", a.len()));
    }
    a.iter().map(number).collect()
}
fn linear(v: &Value, rows: usize, cols: usize, weight: &str, bias: &str) -> Result<Linear, String> {
    let a = v[weight].as_array().ok_or("expected weight matrix")?;
    if a.len() != rows {
        return Err(format!("matrix rows {}, expected {rows}", a.len()));
    }
    let mut data = Vec::with_capacity(rows * cols);
    for row in a {
        data.extend(vector(row, cols)?);
    }
    Ok(Linear {
        rows,
        cols,
        weight: data,
        bias: vector(&v[bias], rows)?,
    })
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = [0.; 4];
    let mut chunks = a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0.iter());
    for (x, y) in &mut chunks {
        s[0] += x[0] * y[0];
        s[1] += x[1] * y[1];
        s[2] += x[2] * y[2];
        s[3] += x[3] * y[3];
    }
    let n = a.len() / 4 * 4;
    let mut result = s.iter().sum::<f32>();
    for i in n..a.len() {
        result += a[i] * b[i];
    }
    result
}
impl Linear {
    fn apply_relu(&self, input: &[f32], out: &mut [f32]) {
        debug_assert_eq!(input.len(), self.cols);
        for (r, value) in out.iter_mut().enumerate().take(self.rows) {
            *value = (self.bias[r] + dot(&self.weight[r * self.cols..(r + 1) * self.cols], input))
                .max(0.);
        }
    }
}
impl Model {
    pub fn load(path: &Path) -> Result<Self, String> {
        let v: Value = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if v["format"] != "jpeg2-channel-policy-f32-v1" || v["dim"] != DIM {
            return Err("unsupported model format or feature dimension".into());
        }
        if v["feature_schema_sha256"].as_str()
            != Some(crate::features::schema_fingerprint().as_str())
        {
            return Err("model feature schema fingerprint mismatch or missing".into());
        }
        let width = v["projection_width"]
            .as_u64()
            .ok_or("missing projection_width")? as usize;
        if width == 0 || width > 1024 {
            return Err("invalid projection width".into());
        }
        let mut groups = Vec::new();
        for c in v["channels"].as_array().ok_or("missing model channels")? {
            let start = c["start"].as_u64().ok_or("channel start")? as usize;
            let end = c["end"].as_u64().ok_or("channel end")? as usize;
            if start >= end || end > DIM {
                return Err("invalid channel range".into());
            }
            groups.push(Group {
                name: c["name"].as_str().ok_or("channel name")?.to_string(),
                start,
                end,
            });
        }
        if groups.is_empty() {
            return Err("empty channel set".into());
        }
        let p = v["projections"].as_array().ok_or("missing projections")?;
        if p.len() != groups.len() {
            return Err("projection/channel count mismatch".into());
        }
        let projections = p
            .iter()
            .zip(&groups)
            .map(|(p, g)| linear(p, width, g.end - g.start, "weight", "bias"))
            .collect::<Result<Vec<_>, _>>()?;
        let h = width * groups.len();
        let base = vector(&v["base"]["weight"], h)?;
        let base_bias = number(&v["base"]["bias"])?;
        let norm = &v["normalization"];
        let mean = vector(&norm["mean"], DIM)?;
        let inverse_std = vector(&norm["inverse_std"], DIM)?;
        if inverse_std.iter().any(|x| *x < 0.) {
            return Err("negative inverse_std".into());
        }
        let clip = number(&norm["clip"])?;
        if clip <= 0. {
            return Err("non-positive normalization clip".into());
        }
        let mut pairs = Vec::new();
        for pair in v["interaction_pairs"]
            .as_array()
            .ok_or("interaction_pairs missing")?
        {
            let p = pair.as_array().ok_or("invalid interaction pair")?;
            if p.len() != 2 {
                return Err("invalid interaction pair length".into());
            }
            let a = p[0].as_u64().ok_or("pair source")? as usize;
            let b = p[1].as_u64().ok_or("pair destination")? as usize;
            if a >= groups.len() || b >= groups.len() {
                return Err("pair channel out of bounds".into());
            }
            pairs.push((a, b));
        }
        let residual = if v["residual"]["enabled"].as_bool().unwrap_or(true) {
            let hidden = v["residual"]["bias1"]
                .as_array()
                .ok_or("residual bias1")?
                .len();
            if hidden == 0 || hidden > 4096 {
                return Err("invalid residual hidden width".into());
            }
            let layer = linear(
                &v["residual"],
                hidden,
                h + pairs.len() * width,
                "weight1",
                "bias1",
            )?;
            Some((
                layer,
                vector(&v["residual"]["weight2"], hidden)?,
                number(&v["residual"]["bias2"])?,
            ))
        } else {
            None
        };
        Ok(Self {
            name: v["name"]
                .as_str()
                .unwrap_or("unnamed direct channel model")
                .into(),
            groups,
            width,
            mean,
            inverse_std,
            clip,
            projections,
            base,
            base_bias,
            pairs,
            residual,
        })
    }
    pub fn score(&self, rows: &[[f32; DIM]], controls: &Controls) -> Scores {
        let h = self.width * self.groups.len();
        let input_len = h + self.pairs.len() * self.width;
        let mut normalized = [0.; DIM];
        let mut hidden = vec![0.; h];
        let mut interaction_input = vec![0.; input_len];
        let mut residual_hidden = vec![0.; self.residual.as_ref().map_or(0, |(l, _, _)| l.rows)];
        let mut logits = Vec::with_capacity(rows.len());
        let mut base_logits = Vec::with_capacity(rows.len());
        for row in rows {
            for i in 0..DIM {
                normalized[i] =
                    ((row[i] - self.mean[i]) * self.inverse_std[i]).clamp(-self.clip, self.clip);
            }
            for (i, g) in self.groups.iter().enumerate() {
                let output = &mut hidden[i * self.width..(i + 1) * self.width];
                if controls.omit_channel.as_deref() == Some(g.name.as_str()) {
                    output.fill(0.);
                } else {
                    self.projections[i].apply_relu(&normalized[g.start..g.end], output);
                }
            }
            let base = self.base_bias + dot(&self.base, &hidden);
            base_logits.push(base);
            let mut value = base;
            if !controls.base_only
                && let Some((layer, weight, bias)) = &self.residual
            {
                interaction_input[..h].copy_from_slice(&hidden);
                for (i, &(a, b)) in self.pairs.iter().enumerate() {
                    for d in 0..self.width {
                        interaction_input[h + i * self.width + d] = if controls.pair_products_off {
                            0.
                        } else {
                            hidden[a * self.width + d] * hidden[b * self.width + d]
                        };
                    }
                }
                layer.apply_relu(&interaction_input, &mut residual_hidden);
                value += bias + dot(weight, &residual_hidden);
            }
            logits.push(value);
        }
        let base_best = argmax(&base_logits);
        let full_best = argmax(&logits);
        Scores {
            logits,
            base_logits,
            base_best,
            full_best,
            interaction_terms: if controls.base_only
                || controls.pair_products_off
                || self.residual.is_none()
            {
                0
            } else {
                rows.len() * self.pairs.len() * self.width
            },
        }
    }
}
pub fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn channel_interaction_changes_decision_and_independent_additive_layout_loads() {
        let mut value = json!({"format":"jpeg2-channel-policy-f32-v1","feature_schema_sha256":crate::features::schema_fingerprint(),"dim":DIM,"channels":[{"name":"a","start":0,"end":1},{"name":"b","start":1,"end":2}],"normalization":{"mean":vec![0.;DIM],"inverse_std":vec![1.;DIM],"clip":8.},"projection_width":1,"projections":[{"weight":[[1.]],"bias":[0.]},{"weight":[[1.]],"bias":[0.]}],"base":{"weight":[1.,0.],"bias":0.},"interaction_pairs":[[0,1]],"residual":{"enabled":true,"weight1":[[0.,0.,1.]],"bias1":[0.],"weight2":[1.],"bias2":0.}});
        let path =
            std::env::temp_dir().join(format!("jpeg2-model-test-{}.json", std::process::id()));
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let m = Model::load(&path).unwrap();
        let mut rows = [[0.; DIM]; 2];
        rows[0][0] = 2.;
        rows[1][0] = 1.;
        rows[1][1] = 3.;
        let full = m.score(&rows, &Controls::default());
        assert_eq!(full.base_best, Some(0));
        assert_eq!(full.full_best, Some(1));
        assert_eq!(full.logits, vec![2., 4.]);
        for controls in [
            Controls {
                base_only: true,
                ..Controls::default()
            },
            Controls {
                pair_products_off: true,
                ..Controls::default()
            },
            Controls {
                omit_channel: Some("b".into()),
                ..Controls::default()
            },
        ] {
            assert_eq!(m.score(&rows, &controls).full_best, Some(0));
        }
        value["residual"] =
            json!({"enabled":false,"weight1":[],"bias1":[],"weight2":[],"bias2":0.});
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(
            Model::load(&path)
                .unwrap()
                .score(&rows, &Controls::default())
                .full_best,
            Some(0)
        );
        fs::remove_file(path).unwrap();
    }
}
