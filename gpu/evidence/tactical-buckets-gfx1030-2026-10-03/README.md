# Native tactical bucket API2: actual gfx1030 evidence

Six CPU wrapper tests and seven actual GPU tests passed; fifty oracle row
comparisons include repeated snapshots. Semantic and timing jobs retired exit0;
all owned workers/supervisors independently absent. No GPU fault was injected.

Timing: same pinned binary, ABI1 E6144 / API2 E6144 / API2 bucket512, six balanced
three-arm rounds at B1/B20 and widths32/128/256, FP32 steps4. Complete native
producer/allocation/admission + resident copy + captured inference measured;
reset/expand/model/checkpoint/capture setup separately excluded and recorded.
Untrained random nonzero heads on one repeated opening-child set; no Elo claim.

Native binary bde1bac9bf8cb052b5d056765329c946a68f1a52ac0a90de63b172e221ff3c05
Neural wrapper commit21ea33f; native base4c92070 plus pinned candidate edits.

Raw timing-results.json and analysis.json retain paired samples and medians.
runtime-source.tar.gz contains compiled sources, binary, Python/provider set,
watchdog and analyzer. archive-index.json hashes every regular member.
manifest.json inventories every payload, excluding itself.

Replay numeric analysis after extracting the archive:
python3 analysis-source/gpu/analyze_tactical_buckets.py timing-results.json

See docs/TACTICAL_BUCKETS.md for API ownership, replay and limits.
