Fixed-depth CPU observations only. Default PVS/aspiration remain off.

Raw depth4/depth5 CSV, declaration, exact runtime commands and source hashes
are retained. analyze.py byte-exactly reproduces summary.json. Binary artifacts
are pinned by absolute path and hash rather than duplicated in this small record.
Validation.json explicitly identifies compact observed test/build results; full
test terminal stdout was not archived. See docs/CLASSICAL_SEARCH.md.
