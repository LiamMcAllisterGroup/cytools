#!/usr/bin/env python3
"""Benchmark the weighted-simplex target case used for performance work.

The harness times the requested phases separately:

* ``Polytope(...).dual()``
* ``triangulate()``
* ``fan.intersection_numbers(symmetrize=False)``
* ``fan.mori_rays()``

It prints both a human-readable summary and a JSON record for auditability.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

from cytools import Polytope

DEFAULT_WEIGHTS = (1, 1, 2, 5, 10)


@dataclass(frozen=True)
class RunResult:
    run: int
    points: int
    simplices: int
    dual_s: float
    triangulate_s: float
    heights_s: float | None
    intersection_numbers_s: float
    mori_rays_s: float
    kappa_entries: int
    mori_rows: int
    construction_audit: dict[str, object]
    total_s: float


def build_polytope(weights: tuple[int, ...]) -> Polytope:
    verts = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [-weights[0], -weights[1], -weights[2], -weights[3], -weights[4]],
    ]
    return Polytope(verts)


def time_once(
    weights: tuple[int, ...], backend: str, *, include_heights: bool
) -> RunResult:
    t0 = time.perf_counter()
    poly = build_polytope(weights)
    dual = poly.dual()
    t1 = time.perf_counter()
    tri = dual.triangulate(backend=backend)
    t2 = time.perf_counter()
    heights_s = None
    if include_heights:
        tri.heights()
        t_heights = time.perf_counter()
        heights_s = t_heights - t2
    else:
        t_heights = t2
    fan = tri.fan()
    intnums = fan.intersection_numbers(symmetrize=False)
    t3 = time.perf_counter()
    mori = fan.mori_rays()
    t4 = time.perf_counter()
    return RunResult(
        run=0,
        points=len(dual.points()),
        simplices=len(tri.simplices()),
        dual_s=t1 - t0,
        triangulate_s=t2 - t1,
        heights_s=heights_s,
        intersection_numbers_s=t3 - t_heights,
        mori_rays_s=t4 - t3,
        kappa_entries=len(intnums),
        mori_rows=mori.shape[0],
        construction_audit=tri.construction_audit(),
        total_s=t4 - t0,
    )


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "min_s": min(values),
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "max_s": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cgal", "qhull"), default="cgal")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--include-heights",
        action="store_true",
        help="Also force tri.heights() so deferred height checks appear in the timing.",
    )
    parser.add_argument(
        "--weights",
        type=int,
        nargs=5,
        default=list(DEFAULT_WEIGHTS),
        metavar=("W1", "W2", "W3", "W4", "W5"),
        help="Weights for the weighted simplex seed.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON report after the human-readable summary.",
    )
    args = parser.parse_args()

    weights = tuple(args.weights)
    results: list[RunResult] = []
    for i in range(args.repeats):
        result = time_once(weights, args.backend, include_heights=args.include_heights)
        results.append(
            RunResult(
                run=i + 1,
                points=result.points,
                simplices=result.simplices,
                dual_s=result.dual_s,
                triangulate_s=result.triangulate_s,
                heights_s=result.heights_s,
                intersection_numbers_s=result.intersection_numbers_s,
                mori_rays_s=result.mori_rays_s,
                kappa_entries=result.kappa_entries,
                mori_rows=result.mori_rows,
                construction_audit=result.construction_audit,
                total_s=result.total_s,
            )
        )
        audit = result.construction_audit
        print(
            f"run {i + 1}: points={result.points} simplices={result.simplices} "
            f"dual={result.dual_s:.3f}s triangulate={result.triangulate_s:.3f}s "
            f"{f'heights={result.heights_s:.3f}s ' if result.heights_s is not None else ''}"
            f"intersection_numbers={result.intersection_numbers_s:.3f}s "
            f"mori_rays={result.mori_rays_s:.3f}s total={result.total_s:.3f}s "
            f"kappa={result.kappa_entries} mori_rows={result.mori_rows}"
        )
        print(
            "  audit: "
            f"triangulation_s={audit['triangulation_s']:.3f} "
            f"star_retries={audit['star_retries']} "
            f"height_check_pending={audit['height_check_pending']} "
            f"height_check_ran={audit['height_check_ran']} "
            f"height_check_s={audit['height_check_s']}"
        )

    totals = [r.total_s for r in results]
    summary = {
        "weights": weights,
        "backend": args.backend,
        "include_heights": args.include_heights,
        "repeats": args.repeats,
        "points": results[0].points if results else None,
        "simplices": results[0].simplices if results else None,
        "total": summarize(totals),
        "dual": summarize([r.dual_s for r in results]),
        "triangulate": summarize([r.triangulate_s for r in results]),
        "heights": (
            summarize([r.heights_s for r in results if r.heights_s is not None])
            if args.include_heights
            else None
        ),
        "intersection_numbers": summarize([r.intersection_numbers_s for r in results]),
        "mori_rays": summarize([r.mori_rays_s for r in results]),
    }
    print(
        "summary: "
        f"total_mean={summary['total']['mean_s']:.3f}s "
        f"total_min={summary['total']['min_s']:.3f}s "
        f"total_max={summary['total']['max_s']:.3f}s"
    )
    if args.json:
        payload = {
            "summary": summary,
            "runs": [asdict(r) for r in results],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
