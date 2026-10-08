"""Tracked experiment framework.

Generic experiment tooling (shared runner, CLI) lives here; each specific
experiment gets its own subpackage with its code and documentation, e.g.
:mod:`cheatdetect.experiments.experiment_1` (``brief.md`` + ``files.md``).

Generated study artifacts stay outside the production model/report directories
so an experiment can never overwrite the deployable winner. This package must
never be imported by :mod:`cheatdetect.app`: the deployable API image is built
with ``--no-dev`` and does not ship research-only dependencies such as torch.
"""
