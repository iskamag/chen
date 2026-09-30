# Chen's reading shelf

Bibliography and links checked on 2026-09-30. These are reading suggestions and
primary sources for future experiments, not claims that Chen reproduces their
results. Publisher pages establish book metadata and contents; whole books have
not been read or copied into this repository.

## Books worth keeping beside the engine

| Priority | Book | Why read it here |
| --- | --- | --- |
| 1 | David Levy, editor. *Computer Chess Compendium*. Springer, 1988. DOI [10.1007/978-1-4757-1968-0](https://link.springer.com/book/10.1007/978-1-4757-1968-0). | A collection of original computer-chess papers. Start with Shannon's “Programming a Computer for Playing Chess,” then the Chess 4.5 chapter, and compare their decomposition with our board/search/evaluation boundaries. The publisher lists the individual chapters. |
| 2 | Donald E. Knuth. *Selected Papers on Analysis of Algorithms*. CSLI Lecture Notes 102, 2000. ISBN 1-57586-212-3. [Author's page and errata](https://www-cs.stanford.edu/~knuth/aa.html). | Chapter 9 contains the alpha-beta analysis. Read alongside node-count experiments; the surrounding essays are useful for asking what a measurement actually establishes. |
| 3 | Stuart Russell and Peter Norvig. *Artificial Intelligence: A Modern Approach*, fourth US edition. [Authors' site](https://aima.cs.berkeley.edu/). | Chapter 5, “Adversarial Search and Games,” supplies the general search vocabulary and independent pseudocode. A broad theory reference rather than a modern engine optimization recipe. |
| 4 | Feng-hsiung Hsu. *Behind Deep Blue: Building the Computer That Defeated the World Chess Champion*. Princeton University Press, 2022 edition. eISBN 978-0-691-23514-1. [Publisher's JSTOR edition](https://www.jstor.org/stable/j.ctv22pzxz1). | A first-person engineering history from Deep Blue's system architect. Useful perspective for hardware/software experiments and development tradeoffs; follow technical claims back to papers before implementing them. |

## Search and experimental method

- Donald E. Knuth and Ronald W. Moore, “An analysis of alpha-beta pruning,”
  *Artificial Intelligence* 6(4), 293–326, 1975.
  [Publisher and DOI](https://www.sciencedirect.com/science/article/pii/0004370275900193).
  Establishes correctness and analyzes how ordering changes work. Our proposed
  use: compare exhaustive minimax with alpha-beta on tiny synthetic trees before
  interpreting chess-position node counts.
- Aske Plaat, *Research Re: Search & Re-Search*, doctoral dissertation,
  Erasmus University Rotterdam, 1996.
  [Author's university-hosted PDF](https://liacs.leidenuniv.nl/~plaata1/papers/thesisg.pdf).
  Read for memory-enhanced search and comparisons of alpha-beta, SSS*, and MTD(f).
  This belongs after a correct transposition table; zero-window methods must be
  compared with identical evaluation, move ordering, and memory budgets.
- [Fishtest mathematics](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)
  documents Stockfish's pentanomial match model, generalized SPRT, normalized Elo,
  and SPSA tuning. Keep the statistical model explicit when comparing runs.
- [Fishtest FAQ](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-FAQ.html)
  discusses selection bias, uncertainty, and hardware sensitivity. In particular,
  the estimated gains of passing patches do not simply add up to an unbiased
  estimate of total progress.
- [Fastchess](https://github.com/Disservin/fastchess) is an external UCI match runner,
  not an engine dependency. Its [manual](https://github.com/Disservin/fastchess/blob/master/man.md)
  specifies paired rounds, opening seeds, time controls, SPRT model selection,
  PGN output, and engine options. Pin its version for reproducible tests.

## The future neural layer

- [Stockfish NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
  explains sparse features, incremental first-layer accumulators, feature
  transformations, quantization, and CPU inference. Use this to design an
  independent evaluator and its correctness oracle; architecture and parameter
  file formats must be versioned together.
- [Stockfish's NNUE trainer](https://github.com/official-stockfish/nnue-pytorch)
  supplies the project's training and serialization implementation. It is a
  reference for the training/runtime boundary, not a dependency of the baseline.
  Any future use of external code, weights, or datasets needs its own provenance
  and applicable license recorded with the experiment.

The baseline uses classical evaluation. Incremental neural inference, training,
SIMD kernels, SPSA, selective pruning, parallel search, and engine tournaments
are future work. The concrete experiment protocol lives in
[EXPERIMENTS.md](EXPERIMENTS.md).
