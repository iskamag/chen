#!/usr/bin/env python3
"""Cold-bind a trained artifact to the confirmed, unchanged feature schema.

No weights or normalization values are changed. This is needed only for the
first fit, whose exporter manifest acquired its schema fingerprint during fit.
Subsequent fits already carry the fingerprint at training ingress.
"""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--features", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
model = json.loads(args.model.read_text())
manifest = json.loads((args.features / "manifest.json").read_text())
assert len(manifest["schema_sha256"]) == 64
assert model["dim"] == manifest["dim"] == len(model["normalization"]["mean"]) == len(model["normalization"]["inverse_std"])
assert model.get("feature_schema_sha256") in [None, manifest["schema_sha256"]]
if model["architecture"] != "flat":
    assert model["channels"] == manifest["channels"]
model["feature_schema_sha256"] = manifest["schema_sha256"]
model["metadata"]["feature_schema_sha256"] = manifest["schema_sha256"]
model["metadata"]["cold_schema_binding"] = {
    "original_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
    "confirmed_feature_manifest_sha256": hashlib.sha256((args.features / "manifest.json").read_bytes()).hexdigest(),
    "unchanged_feature_bytes_sha256": hashlib.sha256((args.features / "features.f32").read_bytes()).hexdigest(),
    "unchanged_index_sha256": hashlib.sha256((args.features / "index.jsonl").read_bytes()).hexdigest(),
    "weights_or_normalization_changed": False,
}
args.out.write_text(json.dumps(model, separators=(",", ":")) + "\n")
print(json.dumps({"model": str(args.out), "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(), "feature_schema_sha256": model["feature_schema_sha256"]}))
