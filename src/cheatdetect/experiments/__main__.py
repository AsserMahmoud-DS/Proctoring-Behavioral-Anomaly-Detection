"""Shared experiment CLI: ``python -m cheatdetect.experiments --phase N``.

Experiment 1 is the preprocessing and model feasibility study. Phases:

- ``3``: build and persist the frozen prepared bundle (Phase 3 study bundle).
- ``4``: run the validation-only search, lock winners, freeze the manifest.
- ``5``: repeat the IF/AE finalists at the predefined seeds.
- ``6``: score the locked approaches on the held-out test split.

Torch is imported lazily by the phases that train the AE, so phases 3-4 and
reporting stay runnable without the dev dependencies.
"""

import argparse
import logging
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cheatdetect.experiments",
        description="Tracked experiment runner (Experiment 1 study phases 3-6).",
    )
    parser.add_argument(
        "--phase",
        type=int,
        required=True,
        choices=(3, 4, 5, 6),
        help="3=bundle, 4=search/lock, 5=seed repeats, 6=test reporting",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Phase 4 run directory used by phases 5-6 (default: study/artifacts/phase_04_study)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    from cheatdetect.experiments.experiment_1 import runner

    if args.phase == 3:
        runner.build_study()
        return 0

    study = runner.load_study()
    if args.phase == 4:
        runner.run_search(study, output_dir=args.run_dir)
    elif args.phase == 5:
        runner.run_seed_repeats(study, run_dir=args.run_dir)
    else:
        from cheatdetect.experiments.experiment_1 import reporting

        reporting.run_phase6(study, run_dir=args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
