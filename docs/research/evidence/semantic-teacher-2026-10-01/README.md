# Closed teacher run: negative training admission

One frozen CPU Stockfish attempt completed all 3,456 queries in 119.96 seconds;
both workers exited zero. The [mechanical audit](semantic-teacher-mechanical-audit-2026-10-01.json)
checks every record/hash, query/history/raw EP, command/options, packet decision,
canonical action, normalization interval/sign and family list. The auditor also
reconstructs histories/actions/material and checks frame arithmetic independently.
Four malformed-metadata controls pass. No engine, model or GPU runs during audit.

| Split | Accepted standalone queries | Accepted root queries | Accepted whole families |
| --- | ---: | ---: | ---: |
| Train | 1,013 | 234 | 6 |
| Validation | 235 | 58 | 0 |
| Test | 230 | 60 | 1 |

The frozen whole-family gate requires eight ordinary standalone packets and
one complete all-legal root packet. It rejects the newest complete packet when
that packet carries a bound or mate score, without substituting an older packet.
Standalone rejection comprised 1,583 bound packets and 11 mates; root rejection
comprised nine bound packets and 23 mates. No rejection came from the depth floor.
These are protocol-yield counts, not teacher accuracy or move-quality metrics.

**The empty validation split prevents training/model-selection admission.**
Individual accepted queries do not override the whole-family gate. No model,
material-scale fit or GPU corpus was created; no score-quality metrics were
opened. No rejected family was replaced, gate relaxed, or teacher query retried.
All 17,128 planned inputs, including rejected families, are recorded in
[spent-inputs.json](spent-inputs.json) for exclusion from future fresh studies.

`raw-records.tar.gz` contains all 3,456 query records with raw transcripts,
declaration, report and worker setups: 3,460 immutable JSON payloads. Every query
SHA in the [producer report](report.json) was independently verified after
archiving; metadata copies also match exactly. The [manifest](manifest.json)
binds this archive, the auditor/negative-control source and logs, structural
review, and original staged result-document snapshot. The [producer archive](../semantic-producer-2026-10-01/manifest.json)
preserves the sealed source/tests/runtime/declaration and the [input replay](../semantic-admission-replay-2026-10-01/README.md)
preserves the independently checked game admission. A different label protocol
requires a separate fresh study; this attempt remains closed.
