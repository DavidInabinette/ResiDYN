from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from src.mpnn import run_mpnn
from src.prepare import run_prepare
from src.selftest import run_demo, run_self_test
from src.validate import run_validation


class _ActivityIndicator:
    """Show that a long-running command is still active without polluting stdout."""

    _FRAMES = ("|", "/", "-", "\\")

    def __init__(self, command: str) -> None:
        self.command = command
        self.started_at = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._interactive = sys.stderr.isatty()

    def __enter__(self) -> "_ActivityIndicator":
        self.started_at = time.monotonic()
        print(f"[{self.command}] Running...", file=sys.stderr, flush=True)
        if self._interactive:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def _spin(self) -> None:
        frame = 0
        while not self._stop.wait(0.2):
            elapsed = int(time.monotonic() - self.started_at)
            print(
                f"\r[{self.command}] Running {self._FRAMES[frame % len(self._FRAMES)]} "
                f"{elapsed}s",
                end="",
                file=sys.stderr,
                flush=True,
            )
            frame += 1

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            print("\r" + " " * 80 + "\r", end="", file=sys.stderr, flush=True)
        elapsed = time.monotonic() - self.started_at
        status = "Failed" if exc_type is not None else "Complete"
        print(f"[{self.command}] {status} ({elapsed:.1f}s)", file=sys.stderr, flush=True)


def _prepare_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prepare",
        help="Create the Phase 1 source of truth from a PDB ID and chain.",
    )
    parser.add_argument("pdb_id", help="Four-character PDB ID, for example 1BTL.")
    parser.add_argument("chain", help="Author or label chain identifier.")
    parser.add_argument("--dms", type=Path, help="Optional ProteinGym-style DMS CSV.")
    parser.add_argument("--target-fasta", type=Path, help="Optional verified DMS WT FASTA.")
    parser.add_argument(
        "--features",
        type=Path,
        help=(
            "Optional residue CSV keyed by target_position, for example ConSurf, "
            "SASA, or secondary structure."
        ),
    )
    parser.add_argument("--assay-id", help="Assay label recorded in provenance.")
    parser.add_argument("--cif", type=Path, help="Use a local mmCIF instead of downloading.")
    parser.add_argument("--assembly", help="RCSB biological assembly number.")
    parser.add_argument("--model", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("Output"))
    parser.add_argument("--mutation-column", default="mutant")
    parser.add_argument("--sequence-column", default="mutated_sequence")
    parser.add_argument("--score-column", default="DMS_score")
    parser.add_argument("--skip-sifts", action="store_true")
    parser.add_argument("--skip-dfi", action="store_true")
    parser.add_argument("--directions", type=int, default=256)
    parser.add_argument("--rotation-trials", type=int, default=8)


def _mpnn_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "mpnn", help="Import ProteinMPNN conditional probabilities."
    )
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--command-file", type=Path, required=True)
    parser.add_argument("--alphabet", default="ACDEFGHIKLMNPQRSTVWYX")


def _validation_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "validate", help="Run paired residue-grouped M0/M1 validation."
    )
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--minimum-residues-per-fold", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260724)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resiDYN.py",
        description="Prepare and validate ResiDYN targets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _prepare_parser(subparsers)
    _mpnn_parser(subparsers)
    _validation_parser(subparsers)
    subparsers.add_parser("self-test", help="Run focused internal tests.")
    demo = subparsers.add_parser("demo", help="Run a synthetic end-to-end smoke test.")
    demo.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with _ActivityIndicator(args.command):
            if args.command == "prepare":
                result = run_prepare(args)
            elif args.command == "mpnn":
                result = run_mpnn(args)
            elif args.command == "validate":
                result = run_validation(args)
            elif args.command == "self-test":
                result = run_self_test()
            else:
                result = run_demo(args.output)
    except (ValueError, OSError) as exc:
        print(
            json.dumps(
                {"status": "failed", "error": type(exc).__name__, "message": str(exc)},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
