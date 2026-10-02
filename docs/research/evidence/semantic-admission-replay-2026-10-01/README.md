# Independent semantic input admission replay

The auditor imports python-chess, not the neural producer or Torch. It reparses
all 384 selected PGN byte ranges, checks source hashes and eligible headers,
replays every observation's full move prefix, raw en passant FEN and previous
position hashes, and reconstructs schema1 identity square by square. All 3,072
observations and 14,056 legal root children agree with the frozen admission.
The 256/64/64 game-family splits have 11,345/2,969/2,814 distinct planned inputs.
Their inputs have zero overlap across families/splits or with the 751,183 keys
independently reconstructed from seven pinned previously used reports.

The two negative controls alter an observation's history or halfmove clock and
recompute its outer source fingerprint. Replay rejects both at the underlying
history/FEN checks. This demonstrates detection beyond file-integrity checking.

`manifest.json` binds the payloads; the reports bind the executed auditor and
negative-control driver. The original decoded PGN and seven old reports are
external inputs, fingerprinted by the replay report and admission. The producer
staging directory contains copies of the seven old reports. To repeat:

```sh
results/qa-venv/bin/python scripts/audit_semantic_admission.py \
  --admission docs/research/evidence/semantic-admission-replay-2026-10-01/admission.json.gz \
  --pgn ../chen-neural/results/semantic-data/lichess_db_standard_rated_2013-01.pgn \
  --spent-repo results/neural-staging \
  --output results/semantic-admission-replay.json
```

This checks input provenance and separation, not independence of players or
opening ancestry. No teacher search, model training, heldout score inspection,
GPU ingestion, timing or playing-strength claim is part of this record.
