# Partial gfx1030 precision/capture diagnostic

The separately reviewed ae2939e attempt ran once and stopped with exit1 after22.104seconds. Square/20261021 completed all five eager/graph/input-copy routes and TRAIN fidelity. Relation capture then rejected a forward-time Pythonconstant→device tensor copy in legacy ChessLatent.attack_edges. No automatic retry, no resume and no final complete.json. Both owned watchdog and process group are retired.

All15square graphs passed exact own-eager CP/latent parity twice, including changed admitted input rows. These checks establish capture mechanics for this checkpoint, not all encoders or transitions. B1INT8graph reverses its eager loss, but FP32graph wins atB32andB256; FP16variants do not win here. All raw balanced timings and numerical errors are retained. Cold setup includeswarmups, synchronization and capture, and is separate from replay. TRAIN fidelity is not teacherquality, modelselection, deployment or Elo.

Original40-model/data/runtime dependencies remain pinned in control/declaration.json and are referenced rather than duplicated. source/ contains this attempt's exact neural closure, native FP16 source/library and tests below. The earlier broad diagnostic/archive remains immutable.
