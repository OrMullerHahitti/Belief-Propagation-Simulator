"""build Min-sum_split_AAMAS-2027_v6_ors_sec6.tex: Roie's v6 with Section 6 replaced by the results-first draft.

the span from the Section 6 heading up to the Conclusions heading is replaced by
SECTION6_results_first_20261006.tex (its header comment dropped), and the two appendix zoom captions are
updated. every anchor must occur exactly once; everything else in v6 is left byte for byte.

usage: uv run python experiments/aamas/section6/build_v6_ors_sec6.py <v6.tex> <out.tex>
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAFT = HERE / "SECTION6_results_first_20261006.tex"
SEC6_START = "\\section{Experimental Evaluation}"
SEC6_END = "\\section{Conclusions}"
OLD_ZOOM = "The panels show DMS, DMS-SCFG, DMS-$k$DS ($k = 1000$), DMS-BDS and MS-SCFG-opt near the end of the run.}"
NEW_ZOOM_RANDOM = (
    "The panels show DMS, DMS-SCFG, DMS-$k$DS ($k = 1000$), DMS-$k^*$DS, DMS-$k^*$DS-MS-MGM and MS-SCFG-opt "
    "near the end of the run, and DABP and DABP-NoSplit, which because of the time stretch show earlier "
    "iterations of their own runs.}"
)
NEW_ZOOM_STRUCTURED = (
    "The panels show DMS, DMS-SCFG, DMS-$k$DS ($k = 1000$), DMS-$k^*$DS and MS-SCFG-opt near the end of the "
    "run, DMS-$k^*$DS-MS-MGM on graph coloring, and DABP and DABP-NoSplit, which because of the time stretch "
    "show earlier iterations of their own runs.}"
)


def once(text: str, needle: str) -> int:
    n = text.count(needle)
    if n != 1:
        raise SystemExit(f"anchor occurs {n} times, expected 1: {needle[:60]}...")
    return text.index(needle)


def main() -> None:
    v6_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    v6 = v6_path.read_text()
    draft = DRAFT.read_text()
    # drop the header comment: the body starts at the section heading (the comment never contains it)
    body = draft[once(draft, SEC6_START) :].rstrip() + "\n\n"

    start, end = once(v6, SEC6_START), once(v6, SEC6_END)
    if not start < end:
        raise SystemExit("Section 6 heading comes after the Conclusions heading")
    new = v6[:start] + body + v6[end:]

    # the two zoom captions carry the same old sentence; the first is the random figure, the second the structured one
    if new.count(OLD_ZOOM) != 2:
        raise SystemExit(f"zoom caption occurs {new.count(OLD_ZOOM)} times, expected 2")
    new = new.replace(OLD_ZOOM, NEW_ZOOM_RANDOM, 1).replace(
        OLD_ZOOM, NEW_ZOOM_STRUCTURED, 1
    )

    out_path.write_text(new)
    before = v6[:start] == new[:start]
    print(
        f"wrote {out_path}: {len(v6.splitlines())} -> {len(new.splitlines())} lines; "
        f"text before Section 6 unchanged: {before}; Section 6 body {len(body.splitlines())} lines"
    )


if __name__ == "__main__":
    main()
