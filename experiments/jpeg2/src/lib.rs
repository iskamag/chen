pub mod features;
pub mod model;
pub mod support;
use features::RootFacts;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};
fn hash(path: &Path) -> String {
    let mut h = Sha256::new();
    let mut f = BufReader::new(File::open(path).unwrap());
    std::io::copy(&mut f, &mut HashWriter(&mut h)).unwrap();
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
struct HashWriter<'a>(&'a mut Sha256);
impl Write for HashWriter<'_> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.update(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn export(input: &str, out: &str, dim: usize) {
    fs::create_dir_all(out).unwrap();
    let output = Path::new(out);
    let featurepath = output.join("features.f32");
    let indexpath = output.join("index.jsonl");
    let mut data = BufWriter::new(File::create(&featurepath).unwrap());
    let mut index = BufWriter::new(File::create(&indexpath).unwrap());
    let mut offset = 0usize;
    let mut positions = 0;
    for line in BufReader::new(File::open(input).unwrap()).lines() {
        let line = line.unwrap();
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut fields = line.splitn(3, '|');
        let id = fields.next().unwrap();
        let fen = fields.next().expect("expected id|fen");
        let p = chen::Position::from_fen(fen).unwrap();
        let root = RootFacts::new(&p);
        let actions = root.actions();
        if let Some(expected) = fields.next() {
            let mut expected: Vec<_> = expected
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            let mut actual: Vec<_> = actions.iter().map(|a| a.mv.to_string()).collect();
            expected.sort();
            actual.sort();
            assert_eq!(actual, expected, "provided legal inventory mismatch {id}");
        }
        let rows = root.feature_rows(&actions, dim).unwrap();
        for row in rows {
            for value in row {
                data.write_all(&value.to_le_bytes()).unwrap();
            }
        }
        writeln!(index,"{}",json!({"id":id,"offset":offset,"count":actions.len(),"moves":actions.iter().map(|a|a.mv.to_string()).collect::<Vec<_>>()})).unwrap();
        offset += actions.len();
        positions += 1;
        if positions % 1000 == 0 {
            eprintln!("exported {positions} positions/{offset} actions");
        }
    }
    data.flush().unwrap();
    index.flush().unwrap();
    let mut manifest = features::schema_for_dimension(dim).unwrap();
    manifest["schema_sha256"] = json!(features::schema_fingerprint_for(dim).unwrap());
    manifest["features_sha256"] = Value::String(hash(&featurepath));
    manifest["index_sha256"] = Value::String(hash(&indexpath));
    manifest["positions"] = json!(positions);
    manifest["actions"] = json!(offset);
    manifest["row_bytes"] = json!(dim * 4);
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    eprintln!("export done {positions} positions/{offset} actions");
}
mod uci;
fn option<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}
fn controls(args: &[String]) -> model::Controls {
    model::Controls {
        base_only: args.iter().any(|a| a == "--base-only"),
        pair_products_off: args.iter().any(|a| a == "--pair-products-off"),
        omit_channel: option(args, "--omit-channel").map(String::from),
    }
}
pub fn diagnose(
    model: &model::Model,
    position: &chen::Position,
    controls: &model::Controls,
) -> Value {
    let started = std::time::Instant::now();
    let root = RootFacts::new(position);
    let actions = root.actions();
    let rows = root
        .feature_rows(&actions, model.dim)
        .expect("validated model feature dimension");
    let encoded = started.elapsed();
    let scores = model.score(&rows, controls);
    let elapsed = started.elapsed();
    json!({"model":model.name,"feature_dim":model.dim,"fen":position.to_fen(),"base_best":scores.base_best.map(|i|actions[i].mv.to_string()),"full_best":scores.full_best.map(|i|actions[i].mv.to_string()),"ranking_changed":scores.base_best!=scores.full_best,"encode_ns":encoded.as_nanos(),"inference_ns":(elapsed-encoded).as_nanos(),"interaction_terms":scores.interaction_terms,"tactical_helper_applied":false,"actions":actions.iter().enumerate().map(|(i,a)|json!({"uci":a.mv.to_string(),"base_logit":scores.base_logits[i],"full_logit":scores.logits[i]})).collect::<Vec<_>>()})
}
pub fn run(args: Vec<String>) -> Result<(), String> {
    let command = args.get(1).map_or("uci", String::as_str);
    match command {
        "export" => {
            if args.len() < 4 {
                return Err("export INPUT.tsv OUTPUTDIR".into());
            }
            export(
                &args[2],
                &args[3],
                if args.iter().any(|a| a == "--support") {
                    features::SUPPORT_DIM
                } else {
                    features::DIM
                },
            );
            Ok(())
        }
        "schema" => {
            let dim = if args.iter().any(|a| a == "--support") {
                features::SUPPORT_DIM
            } else {
                features::DIM
            };
            let mut schema = features::schema_for_dimension(dim).unwrap();
            schema["schema_sha256"] = json!(features::schema_fingerprint_for(dim).unwrap());
            println!("{schema}");
            Ok(())
        }
        "predict" => {
            let path = option(&args, "--model")
                .ok_or("predict --model MODEL --input INPUT.tsv [--out OUTPUT.jsonl]")?;
            let input = option(&args, "--input").ok_or("missing --input")?;
            let model = model::Model::load(Path::new(path))?;
            let ctl = controls(&args);
            let mut output: Box<dyn Write> = if let Some(path) = option(&args, "--out") {
                Box::new(BufWriter::new(
                    File::create(path).map_err(|e| e.to_string())?,
                ))
            } else {
                Box::new(std::io::stdout())
            };
            for line in BufReader::new(File::open(input).map_err(|e| e.to_string())?).lines() {
                let line = line.map_err(|e| e.to_string())?;
                if line.starts_with('#') || line.trim().is_empty() {
                    continue;
                }
                let mut parts = line.splitn(3, '|');
                let id = parts.next().ok_or("missing id")?;
                let fen = parts.next().ok_or("expected id|fen")?;
                let p = chen::Position::from_fen(fen)?;
                let mut record = diagnose(&model, &p, &ctl);
                record["id"] = json!(id);
                writeln!(output, "{record}").map_err(|e| e.to_string())?;
            }
            output.flush().map_err(|e| e.to_string())?;
            Ok(())
        }
        "diagnose" => {
            let model = model::Model::load(Path::new(
                option(&args, "--model").ok_or("missing --model")?,
            ))?;
            let position =
                chen::Position::from_fen(option(&args, "--fen").ok_or("missing --fen")?)?;
            println!("{}", diagnose(&model, &position, &controls(&args)));
            Ok(())
        }
        "uci" => {
            let model_path = option(&args, "--model").ok_or(
                "uci --model MODEL.json [--base-only|--pair-products-off|--omit-channel NAME]",
            )?;
            let model = model::Model::load(Path::new(model_path))?;
            uci::run(model, controls(&args))
        }
        _ => Err("expected export, schema, predict, diagnose, or uci subcommand".into()),
    }
}
