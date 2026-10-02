# Stockfish 19 ordinary score packet protocol

The next strong-label study uses fresh game families and an explicitly different
packet policy. The closed run remains a negative admission result: its strict
newest-packet rule retained 6/0/1 train/validation/test families. Its inputs and
labels will not be reused for this study. Independent review passed 31 CPU
tests and replayed the fresh admission. Root separately sealed the bounded
source/data declaration before starting its one CPU teacher attempt; model
training remains a later milestone.

## Why the final output can be bounded

The pinned [Stockfish 19 search source](https://github.com/official-stockfish/Stockfish/blob/sf_19/src/search.cpp)
sets an upper or lower flag when a root value reaches a search-window boundary,
and reports that boundary through its UCI score. At the end of search, it can
emit updated PV information before `bestmove`. A stopped window can therefore
follow an earlier ordinary iteration. An ordinary packet is a finite selective
search estimate; absence of a bound flag does not establish exact minimax.

`SearchManager::output_pv` emits ranks 1 through K in one loop and uses a shared
node count. Unsearched rows can carry their preceding scores and report one
less depth. The resulting emission can contain mixed depths. These facts
justify recording whole emissions and rejecting mixtures as point-label
packets. The local pinned source SHA256 is
`a934524dd2f386ec38cdf95b7b2e41cecdc85c9b17e12c0668621e6ae16be28a`.
The [UCI formatting source](https://github.com/official-stockfish/Stockfish/blob/sf_19/src/uci.cpp)
preserves the bound flag, with local SHA256
`a31fa10719b1d9ecaa23f774f14a5a995be65eeb52722798c2f2bb4255991b21`.

## Packet selection and rejection

Rank 1 begins an emission. Finalize it at the next rank 1 or end of output,
including any intervening scored rows; do not publish immediately on seeing
rank K and then overlook an extra or duplicate row. Never borrow a rank from
an earlier emission.

| Evidence | Declared action |
| --- | --- |
| Complete ordinary packet at one depth and node count | Eligible if depth is at least six, all PVs are legal, root actions uniquely cover the required support and scores are nonincreasing with ties allowed |
| Clean incomplete or mixed-depth emission | Ineligible; retain its identity in provenance |
| Upper or lower bound emission | Ineligible as an ordinary point target |
| Duplicate, gap or excess rank; malformed scored grammar; illegal PV/support or nonmonotonic ordinary scores | Reject the query |
| Any mate, tablebase hit or reserved decisive CP evidence | Reject the query, including when an earlier ordinary packet exists |

Choose the eligible packet with the highest common reported depth, breaking
ties by its latest completion ordinal. Record its start/completion identity,
depth and node count, the last emission's disposition and the final `bestmove`.
The selected rank-1 PV may differ from that final choice. No final-choice or
uniform-child-horizon equivalence is asserted. Preserve the existing root
material normalization and the distinction between standalone and parent
MultiPV query kinds.

## Fresh study boundaries

Use seed 20261008 and the same 256/64/64 family counts, header filters, mover
schedule and node budgets as the first study. Admission excludes all 751,183
older canonical inputs and all 17,128 planned inputs from the closed run,
including its rejected families. Check exact move-prefix history and every
planned root child before labels. Do not replace families after observing
packet decisions.

Each standalone query requests 32,000 nodes; each complete-root query requests
256,000. Keep two CPU workers, a 1,200-second cooperative job ceiling including
the existing cleanup reserve, and the tested cancellation/reaping/quarantine
rules. Query rejection retains the eight-observation/whole-root family gate.
Partial or faulty attempts admit no families. This study produces CPU UCI label
records; GPU ingestion, material-scale fitting and model training need their
own later declaration. It establishes no playing strength or rating.

The frozen seed-20261008 admission contains 256/64/64 game families, 3,072
observations and 14,012 legal root children. All eight spent-input reports
were replayed: 768,311 excluded canonical identities, with zero overlap.
Source, declaration and raw-history admission are staged under
`results/neural-packet-v2`; the earlier source snapshot remains unchanged.
The [preparation archive](evidence/semantic-packet-v2-preparation-2026-10-01/README.md)
preserves all 16 pinned modules, five fixture modules, source mappings,
independent logs, eight spent reports, admission and runtime fingerprints.
The label declaration SHA256 is
`a6f09aa8f34eaffa097b6602dc301cd00cd0e86f810ed3a357c4ce7daee08bda`.

The one bounded CPU attempt completed all 3,456 queries in 134.96 seconds with
both worker exits zero. Its frozen whole-family rule retained 233/62/55
train/validation/test families. Their standalone rows number 1,864/496/440;
complete-root action rows number 8,477/2,106/2,118. Rejected families are not
replaced. Independent transcript, packet and frame audit passed all 3,456
records, including exact source/runtime/declaration bindings, full histories,
legal support, emission selection and Decimal conversion checks. Four
synthetic auditor test groups pass. These structural checks establish no model
quality; corpus ingestion and learning still require separate declarations.
The [closed teacher archive](evidence/semantic-teacher-packet-v2-2026-10-01/README.md)
preserves the complete transcripts and records, query plan, admission, audit
and all 17,084 planned canonical inputs for future study exclusion.

Among all accepted queries, the selected rank-1 PV differs from the final move
in 84 standalone and 93 complete-root queries. The last emitted packet is
bounded in 1,508 standalone queries and mixed-depth in 367 root queries.
Those dispositions are preserved alongside the earlier selected ordinary
packet. The distinction between historical packet targets and final search
choices therefore matters in the actual records.
