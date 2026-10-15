#!/usr/bin/env python3
"""Small actual MDCT ablation; no chess or optimized-kernel claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def signals(protocol):
    length = protocol["signal_length"]
    for location in range(length):
        for sign in (1.0, -1.0):
            value = np.zeros(length)
            value[location] = sign
            yield f"impulse-{location}-{int(sign)}", value
    locations = np.arange(length)
    for frequency in (1, 3, 7, 17, 31, 63):
        for phase_index, phase in enumerate((0.0, np.pi / 3, np.pi / 2, np.pi)):
            yield f"sine-{frequency}-{phase_index}", np.sin(
                2 * np.pi * frequency * locations / length + phase
            )
    for edge in (1, 7, 8, 15, 16, 31, 32, 127, 128, 255):
        yield f"step-{edge}", np.where(locations < edge, -1.0, 1.0)
    rng = np.random.Generator(np.random.PCG64(protocol["seed"]))
    for number in range(32):
        yield f"normal-{number}", rng.normal(size=length)


def basis(count):
    sample = np.arange(2 * count)
    coefficient = np.arange(count)
    cosine = np.cos(
        np.pi / count
        * (sample[None, :] + 0.5 + count / 2)
        * (coefficient[:, None] + 0.5)
    )
    window = np.sin(np.pi / (2 * count) * (sample + 0.5))
    return cosine, window


def overlap_add(blocks, starts, length):
    output = np.zeros(length)
    for start, block in zip(starts, blocks, strict=True):
        output[start : start + len(block)] += block
    return output


def measure(protocol):
    tolerance = protocol["tolerance_max_absolute"]
    corpus = list(signals(protocol))
    result = {}
    for count in protocol["coefficient_counts"]:
        cosine, window = basis(count)
        metrics = {
            name: {"error_cases": 0, "max_abs_error": 0.0, "relative_l2": []}
            for name in protocol["variants"]
        }
        isolated_alias_max = 0.0
        overlapped_alias_max = 0.0
        gain_aligned = []
        first_example = None
        for identity, original in corpus:
            padded = np.pad(original, (count, count))
            starts = list(range(0, len(padded) - 2 * count + 1, count))
            blocks = np.stack([padded[s : s + 2 * count] for s in starts])
            encoded = (blocks * window) @ cosine.T

            def inverse(coefficients, synthesis_window=window):
                return (2.0 / count) * (coefficients @ cosine) * synthesis_window

            reconstructed_blocks = inverse(encoded)
            aliases = reconstructed_blocks - blocks * window**2
            isolated_alias_max = max(isolated_alias_max, float(np.max(np.abs(aliases))))
            alias_sum = overlap_add(aliases, starts, len(padded))[
                count : count + len(original)
            ]
            overlapped_alias_max = max(
                overlapped_alias_max, float(np.max(np.abs(alias_sum)))
            )
            polarity = np.where(np.arange(len(starts)) % 2 == 0, 1.0, -1.0)
            variants = {
                "signed_matched_overlap_add": reconstructed_blocks,
                "absolute_coefficients_matched_overlap_add": inverse(np.abs(encoded)),
                "signed_mismatched_synthesis_window": inverse(encoded, np.ones(2 * count)),
                "signed_alternate_block_polarity": inverse(encoded * polarity[:, None]),
                "signed_quantized_step_0.25": inverse(np.rint(encoded / 0.25) * 0.25),
            }
            for name, decoded_blocks in variants.items():
                decoded = overlap_add(decoded_blocks, starts, len(padded))[
                    count : count + len(original)
                ]
                maximum = float(np.max(np.abs(decoded - original)))
                row = metrics[name]
                row["max_abs_error"] = max(row["max_abs_error"], maximum)
                row["error_cases"] += int(maximum > tolerance)
                row["relative_l2"].append(
                    float(np.linalg.norm(decoded - original) / np.linalg.norm(original))
                )
                if name == "signed_mismatched_synthesis_window":
                    gain = float(np.dot(decoded, original) / np.dot(decoded, decoded))
                    gain_aligned.append(
                        float(np.linalg.norm(gain * decoded - original) / np.linalg.norm(original))
                    )
            if identity == "impulse-8-1":
                opposite = (-blocks * window) @ cosine.T
                first_example = {
                    "input": identity,
                    "opposite_input": "impulse-8--1",
                    "max_abs_magnitude_difference": float(
                        np.max(np.abs(np.abs(encoded) - np.abs(opposite)))
                    ),
                    "different_original_signal_l2": float(2 * np.linalg.norm(original)),
                    "identical_absolute_coefficient_decodes": bool(
                        np.array_equal(inverse(np.abs(encoded)), inverse(np.abs(opposite)))
                    ),
                    "nonzero_encoded_rows": [
                        {"block_start": starts[i], "coefficients": row.tolist()}
                        for i, row in enumerate(encoded)
                        if np.max(np.abs(row)) > tolerance
                    ],
                }
        for row in metrics.values():
            row["mean_relative_l2_error"] = float(np.mean(row.pop("relative_l2")))
        assert metrics["signed_matched_overlap_add"]["error_cases"] == 0
        assert overlapped_alias_max < tolerance
        assert isolated_alias_max > 0.1
        assert first_example["identical_absolute_coefficient_decodes"]
        result[str(count)] = {
            "cases": len(corpus),
            "analysis_block_samples": 2 * count,
            "coefficients_per_block": count,
            "analysis_matrix_rank": int(np.linalg.matrix_rank(cosine * window)),
            "overlap_hop": count,
            "window_complementarity_max_error": float(
                np.max(np.abs(window[:count] ** 2 + window[count:] ** 2 - 1))
            ),
            "maximum_isolated_block_alias_abs": isolated_alias_max,
            "maximum_overlapped_alias_abs": overlapped_alias_max,
            "mismatched_window_mean_gain_aligned_relative_l2": float(np.mean(gain_aligned)),
            "variants": metrics,
            "magnitude_collision": first_example,
        }
    return {
        "experiment": "actual-mdct-alias-cancellation",
        "protocol_sha256": hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "results": result,
        "limits": protocol["limits"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    result = measure(protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.out), "N": {
        key: {"cases": row["cases"], "max_reconstruction_error":
              row["variants"]["signed_matched_overlap_add"]["max_abs_error"],
              "overlapped_alias": row["maximum_overlapped_alias_abs"]}
        for key, row in result["results"].items()
    }}, allow_nan=False))


if __name__ == "__main__":
    main()
