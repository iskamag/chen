# Hybrid versus stock Chen: no demonstrated improvement

Hybrid v0 scored **5 wins, 4 draws, 7 losses (7/16 points)** against stock Chen
at 500 ms per searched move. Eight book seeds, 101–108, were fixed before play;
each was played with colors swapped. Both sides used the same controller book,
up to twelve plies, one CPU thread and 16 MiB hash. Stock used the web Serious
configuration: Tapered evaluation, PVS, TT, quiescence and move ordering, with
aspiration disabled. Hybrid retained its original 25 cp aspiration window and
35 ms internal margin. These are whole-bot settings, not an isolated JPEG ablation.

The result provides no evidence of a strength gain. Sixteen games cannot exclude
a modest improvement or regression. No rating or equivalence claim follows.

Decision telemetry is more revealing: **zero of 813 conventional search decisions
were overridden by the JPEG gate**. Another five decisions were immediate-mate
shortcuts. The prepass found alternatives allowing an opposing mate-in-one in
64 positions, but the search already avoided those alternatives or returned the
same move. Detecting a danger is not evidence that the gate improved the choice.

In **32 identical-position probes**, Hybrid and stock chose the same move
**32/32 times**. The positions were the eight book endpoints and plies 24, 40 and
56 of each candidate-White game, selected by rule rather than result. Every
probe began with a fresh TT and full game history. A supplementary stock control
with aspiration 25 and a 465 ms budget agreed with Hybrid 31/32 times; an
identical repeat of that control agreed 32/32 times. That control matches the
margin and aspiration but does not subtract the variable prepass cost.

There is no observed independent playstyle here: the evaluation and continuation
search are shared. Same-position captures, checks, pawn moves and castling also
match exactly between stock and Hybrid. Differences in complete games can arise
from the time budget, aspiration and subsequent divergent positions. The current
Hybrid is a tactical safety wrapper; it does not deliver the broader compressed
continuation architecture or establish a useful improvement over stock Chen.

All sixteen games and 1,804 legal plies pass separate rule/PGN/JSON/FEN, book and
executable audits. Median searched-move wall times were 465.8 ms for Hybrid and
500.8 ms for stock. The five collector tests pass, including explicit distinction
between the searched proposal and the played move.

Raw collection ran in RAM. Only compact results, named [PGNs](../../results/jpeg-vs-stock-20261015/games.pgn)
and compressed audit evidence are retained under `results/jpeg-vs-stock-20261015/`;
no videos, audio, frames or copied engines were created. Compact committed
[evidence](evidence/jpeg-vs-stock-20261015.json) includes artifact hashes.
