from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import TargetConfig, resolve_path
from .io_utils import build_manifest, write_csv, write_json


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}
BACKBONE = {"N", "CA", "C", "O"}


class StructureError(ValueError):
    """Raised when structure parsing or sequence mapping is unsafe."""


@dataclass(frozen=True)
class Atom:
    group: str
    atom_name: str
    altloc: str
    residue_name: str
    label_asym_id: str
    label_seq_id: int
    auth_asym_id: str
    auth_seq_id: str
    insertion_code: str
    x: float
    y: float
    z: float
    occupancy: float
    element: str
    model_number: int


def _clean_cif(value: str) -> str:
    return "" if value in {".", "?"} else value


def _as_int(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise StructureError(f"Expected integer {field}, got {value!r}") from exc


def _as_float(value: str, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cif_tokens(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lexer = shlex.shlex(text, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    return list(lexer)


def parse_mmcif_atom_site(path: str | Path) -> list[Atom]:
    """Read the `_atom_site` loop from an mmCIF file.

    This intentionally small parser handles quoted atom-site values and row
    wrapping. Semicolon-delimited multiline values are not valid atom-site
    scalar fields and are not supported.
    """

    source = Path(path)
    tokens = _cif_tokens(source)
    index = 0
    while index < len(tokens):
        if tokens[index].lower() != "loop_":
            index += 1
            continue
        index += 1
        tags: list[str] = []
        while index < len(tokens) and tokens[index].startswith("_"):
            tags.append(tokens[index])
            index += 1
        if not tags or not all(tag.startswith("_atom_site.") for tag in tags):
            while index < len(tokens) and tokens[index].lower() != "loop_":
                if tokens[index].startswith("_"):
                    break
                index += 1
            continue

        rows: list[list[str]] = []
        while index < len(tokens):
            token = tokens[index]
            if token.lower() == "loop_" or token.lower().startswith(("data_", "save_")):
                break
            if token.startswith("_") and len(rows) * len(tags) == 0:
                break
            if index + len(tags) > len(tokens):
                raise StructureError("Incomplete _atom_site row at end of mmCIF")
            row = tokens[index:index + len(tags)]
            if row[0].lower() == "loop_" or row[0].startswith("_"):
                break
            rows.append(row)
            index += len(tags)
        return _atoms_from_cif_rows(tags, rows)
    raise StructureError(f"No _atom_site loop found in {source}")


def _atoms_from_cif_rows(tags: list[str], rows: list[list[str]]) -> list[Atom]:
    col = {tag.split(".", 1)[1]: i for i, tag in enumerate(tags)}

    def required(*names: str) -> int:
        for name in names:
            if name in col:
                return col[name]
        raise StructureError(f"mmCIF _atom_site is missing one of: {names}")

    indexes = {
        "group": required("group_PDB"),
        "atom": required("label_atom_id", "auth_atom_id"),
        "resname": required("label_comp_id", "auth_comp_id"),
        "label_chain": required("label_asym_id"),
        "label_seq": required("label_seq_id"),
        "x": required("Cartn_x"),
        "y": required("Cartn_y"),
        "z": required("Cartn_z"),
    }
    atoms: list[Atom] = []
    for row in rows:
        label_seq = _clean_cif(row[indexes["label_seq"]])
        if not label_seq:
            continue
        get = lambda name, default="": row[col[name]] if name in col else default
        atoms.append(
            Atom(
                group=row[indexes["group"]].upper(),
                atom_name=_clean_cif(row[indexes["atom"]]).upper(),
                altloc=_clean_cif(get("label_alt_id")),
                residue_name=_clean_cif(row[indexes["resname"]]).upper(),
                label_asym_id=_clean_cif(row[indexes["label_chain"]]),
                label_seq_id=_as_int(label_seq, "label_seq_id"),
                auth_asym_id=_clean_cif(get("auth_asym_id", row[indexes["label_chain"]])),
                auth_seq_id=_clean_cif(get("auth_seq_id", label_seq)),
                insertion_code=_clean_cif(get("pdbx_PDB_ins_code")),
                x=_as_float(row[indexes["x"]]),
                y=_as_float(row[indexes["y"]]),
                z=_as_float(row[indexes["z"]]),
                occupancy=_as_float(get("occupancy", "1.0"), 1.0),
                element=_clean_cif(get("type_symbol")),
                model_number=_as_int(
                    _clean_cif(get("pdbx_PDB_model_num", "1")) or "1",
                    "pdbx_PDB_model_num",
                ),
            )
        )
    return atoms


def parse_pdb_atoms(path: str | Path, label_asym_id: str | None = None) -> list[Atom]:
    atoms: list[Atom] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21].strip() or "_"
        if label_asym_id is not None and chain != label_asym_id:
            continue
        auth_seq = line[22:26].strip()
        if not auth_seq:
            continue
        atoms.append(
            Atom(
                group=line[0:6].strip(),
                atom_name=line[12:16].strip().upper(),
                altloc=line[16].strip(),
                residue_name=line[17:20].strip().upper(),
                label_asym_id=chain,
                label_seq_id=_as_int(auth_seq, "PDB residue number"),
                auth_asym_id=chain,
                auth_seq_id=auth_seq,
                insertion_code=line[26].strip(),
                x=_as_float(line[30:38]),
                y=_as_float(line[38:46]),
                z=_as_float(line[46:54]),
                occupancy=_as_float(line[54:60], 1.0),
                element=line[76:78].strip(),
                model_number=1,
            )
        )
    if not atoms:
        raise StructureError(f"No atoms found in {path}")
    return atoms


def read_atoms(
    path: str | Path,
    fmt: str,
    label_asym_id: str | None = None,
    model_number: int = 1,
) -> list[Atom]:
    if fmt.lower() in {"mmcif", "cif"}:
        atoms = parse_mmcif_atom_site(path)
    elif fmt.lower() == "pdb":
        atoms = parse_pdb_atoms(path)
    else:
        raise StructureError(f"Unsupported structure format: {fmt}")
    return [
        atom for atom in atoms
        if atom.model_number == model_number
        and (label_asym_id is None or atom.label_asym_id == label_asym_id)
    ]


def structure_inventory(atoms: Iterable[Atom]) -> dict[str, object]:
    atoms = list(atoms)
    residue_keys = {
        (
            atom.group, atom.label_asym_id, atom.label_seq_id,
            atom.residue_name, atom.auth_seq_id, atom.insertion_code,
        )
        for atom in atoms
    }
    residue_names = sorted({key[3] for key in residue_keys})
    nonstandard_polymer = sorted(
        {
            key[3] for key in residue_keys
            if key[0] == "ATOM" and key[3] not in AA3_TO_1
        }
    )
    waters = {"HOH", "WAT", "DOD"}
    hetero = sorted(
        {
            key[3] for key in residue_keys
            if key[0] == "HETATM" and key[3] not in waters
        }
    )
    return {
        "n_atoms": len(atoms),
        "n_residue_instances": len(residue_keys),
        "chains": sorted({atom.label_asym_id for atom in atoms}),
        "residue_names": residue_names,
        "nonstandard_polymer_residue_names": nonstandard_polymer,
        "hetero_residue_names_excluding_water": hetero,
        "water_residue_count": sum(key[3] in waters for key in residue_keys),
        "alternate_location_labels": sorted(
            {atom.altloc for atom in atoms if atom.altloc}
        ),
        "minimum_occupancy": min((atom.occupancy for atom in atoms), default=None),
    }


def select_altlocs(atoms: Iterable[Atom]) -> list[Atom]:
    """Select one coordinate per residue/atom deterministically."""

    grouped: dict[tuple[str, int, str], list[Atom]] = {}
    for atom in atoms:
        key = (atom.label_asym_id, atom.label_seq_id, atom.atom_name)
        grouped.setdefault(key, []).append(atom)

    def tie_rank(altloc: str) -> tuple[int, str]:
        if altloc == "":
            return (0, "")
        if altloc == "A":
            return (1, "A")
        return (2, altloc)

    selected = [
        sorted(group, key=lambda atom: (-atom.occupancy, tie_rank(atom.altloc)))[0]
        for group in grouped.values()
    ]
    return sorted(selected, key=lambda atom: (atom.label_seq_id, atom.atom_name))


def global_sequence_map(observed: str, target: str) -> dict[int, int]:
    """Needleman-Wunsch map from 1-based observed positions to target positions."""

    if not observed or not target:
        raise StructureError("Observed and target sequences must both be non-empty")
    n, m = len(observed), len(target)
    score = np.zeros((n + 1, m + 1), dtype=int)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)
    gap = -2
    score[:, 0] = np.arange(n + 1) * gap
    score[0, :] = np.arange(m + 1) * gap
    trace[1:, 0] = 1
    trace[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = (
                score[i - 1, j - 1] + (2 if observed[i - 1] == target[j - 1] else -1),
                score[i - 1, j] + gap,
                score[i, j - 1] + gap,
            )
            move = int(np.argmax(candidates))
            score[i, j] = candidates[move]
            trace[i, j] = move
    mapping: dict[int, int] = {}
    i, j = n, m
    while i or j:
        move = trace[i, j]
        if i and j and move == 0:
            if observed[i - 1] == target[j - 1]:
                mapping[i] = j
            i -= 1
            j -= 1
        elif i and (j == 0 or move == 1):
            i -= 1
        else:
            j -= 1
    return mapping


def build_residue_table(
    atoms: Iterable[Atom],
    *,
    structure_instance_id: str,
    label_asym_id: str,
    target_sequence: str,
    dms_position_offset: int = 0,
    contact_cutoff: float = 8.0,
) -> pd.DataFrame:
    selected = select_altlocs(atoms)
    residues: dict[int, list[Atom]] = {}
    for atom in selected:
        if atom.label_asym_id != label_asym_id:
            continue
        if atom.residue_name not in AA3_TO_1:
            continue
        residues.setdefault(atom.label_seq_id, []).append(atom)
    if not residues:
        raise StructureError(f"No recognized polymer residues for label_asym_id={label_asym_id}")

    seq_ids = sorted(residues)
    observed_sequence = "".join(AA3_TO_1[residues[seq_id][0].residue_name] for seq_id in seq_ids)
    sequence_map = global_sequence_map(observed_sequence, target_sequence.upper())
    rows: list[dict[str, object]] = []
    for ordinal, seq_id in enumerate(seq_ids, start=1):
        group = residues[seq_id]
        atom_map = {atom.atom_name: atom for atom in group}
        representative = group[0]
        aa = AA3_TO_1[representative.residue_name]
        target_position = sequence_map.get(ordinal)
        identity_ok = (
            target_position is not None
            and target_sequence[target_position - 1].upper() == aa
        )
        ca = atom_map.get("CA")
        rows.append(
            {
                "structure_instance_id": structure_instance_id,
                "label_asym_id": label_asym_id,
                "label_seq_id": seq_id,
                "auth_asym_id": representative.auth_asym_id,
                "auth_seq_id": representative.auth_seq_id,
                "insertion_code": representative.insertion_code,
                "residue_name": representative.residue_name,
                "wt": aa,
                "target_position": target_position if identity_ok else pd.NA,
                "dms_position": (
                    target_position - dms_position_offset if identity_ok else pd.NA
                ),
                "mapping_identity_ok": identity_ok,
                "has_n": "N" in atom_map,
                "has_ca": ca is not None,
                "has_c": "C" in atom_map,
                "has_o": "O" in atom_map,
                "ca_x": ca.x if ca else np.nan,
                "ca_y": ca.y if ca else np.nan,
                "ca_z": ca.z if ca else np.nan,
                "mean_occupancy": float(np.mean([atom.occupancy for atom in group])),
                "selected_altlocs": ",".join(sorted({atom.altloc or "." for atom in group})),
            }
        )
    frame = pd.DataFrame(rows)
    frame["dfi_context_mask"] = frame["has_ca"]
    frame["analysis_mask"] = (
        frame[["has_n", "has_ca", "has_c", "has_o"]].all(axis=1)
        & frame["mapping_identity_ok"]
    )
    completeness = frame[["has_n", "has_ca", "has_c", "has_o"]].sum(axis=1) / 4.0
    frame["coordinate_quality"] = completeness * frame["mean_occupancy"].clip(0.0, 1.0)

    mapped = set(frame.loc[frame["mapping_identity_ok"], "target_position"].dropna().astype(int))
    frame["gap_adjacent"] = frame["target_position"].apply(
        lambda value: int(
            pd.notna(value)
            and (
                (int(value) > 1 and int(value) - 1 not in mapped)
                or (int(value) < len(target_sequence) and int(value) + 1 not in mapped)
            )
        )
    )
    frame["contact_number"] = _contact_numbers(frame, contact_cutoff)
    frame["dfi_serial"] = pd.NA
    context_indices = frame.index[frame["dfi_context_mask"]]
    frame.loc[context_indices, "dfi_serial"] = np.arange(1, len(context_indices) + 1)
    for column in ("target_position", "dms_position", "dfi_serial"):
        frame[column] = frame[column].astype("Int64")
    return frame


def _contact_numbers(frame: pd.DataFrame, cutoff: float) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = frame["has_ca"].to_numpy(bool)
    coords = frame.loc[valid, ["ca_x", "ca_y", "ca_z"]].to_numpy(float)
    if len(coords):
        distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        counts = ((distances <= cutoff) & (distances > 0)).sum(axis=1)
        result.loc[valid] = counts
    return result


def write_minimal_pdb(
    atoms: Iterable[Atom],
    residue_table: pd.DataFrame,
    path: str | Path,
    *,
    mask_column: str,
    ca_only: bool,
) -> Path:
    selected = select_altlocs(atoms)
    allowed = set(
        residue_table.loc[residue_table[mask_column], "label_seq_id"].astype(int)
    )
    serial_map = (
        residue_table.set_index("label_seq_id")["dfi_serial"].to_dict()
        if ca_only else {}
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    serial = 1
    for atom in selected:
        if atom.label_seq_id not in allowed:
            continue
        if ca_only and atom.atom_name != "CA":
            continue
        if not ca_only and atom.atom_name not in BACKBONE:
            continue
        atom_serial = int(serial_map[atom.label_seq_id]) if ca_only else serial
        chain = (atom.auth_asym_id or atom.label_asym_id or "A")[0]
        auth_seq = int(re.sub(r"[^0-9-]", "", atom.auth_seq_id) or atom.label_seq_id)
        lines.append(
            f"ATOM  {atom_serial:5d} {atom.atom_name:^4s} {atom.residue_name:>3s} "
            f"{chain}{auth_seq:4d}{(atom.insertion_code or ' '):1s}   "
            f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
            f"{atom.occupancy:6.2f}{0.0:6.2f}          {atom.element:>2s}"
        )
        serial += 1
    lines.extend(["TER", "END"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def run_structure_stage(config: TargetConfig) -> dict[str, Path]:
    source = resolve_path(config, config.require("structure.path"))
    fmt = config.get("structure.format", source.suffix.lstrip("."))
    chain = str(config.require("structure.label_asym_id"))
    model_number = int(config.get("structure.source_model_number", 1))
    all_atoms = read_atoms(source, fmt, model_number=model_number)
    atoms = [atom for atom in all_atoms if atom.label_asym_id == chain]
    if not atoms:
        raise StructureError(
            f"No atoms found for label_asym_id={chain!r}, model={model_number}"
        )
    table = build_residue_table(
        atoms,
        structure_instance_id=str(config.require("project.structure_instance_id")),
        label_asym_id=chain,
        target_sequence=str(config.require("structure.target_sequence")).upper(),
        dms_position_offset=int(config.get("structure.dms_position_offset", 0)),
        contact_cutoff=float(config.get("structure.contact_cutoff_angstrom", 8.0)),
    )
    output_dir = config.output_dir
    outputs = {
        "residues": write_csv(table, output_dir / "residues.csv"),
        "dfi_pdb": write_minimal_pdb(
            atoms, table, output_dir / "dfi_input.pdb",
            mask_column="dfi_context_mask", ca_only=True,
        ),
        "mpnn_pdb": write_minimal_pdb(
            atoms, table, output_dir / "proteinmpnn_input.pdb",
            mask_column="analysis_mask", ca_only=False,
        ),
        "serial_map": write_csv(
            table.loc[table["dfi_context_mask"], [
                "structure_instance_id", "label_asym_id", "label_seq_id",
                "target_position", "auth_asym_id", "auth_seq_id",
                "insertion_code", "wt", "dfi_serial",
            ]],
            output_dir / "dfi_serial_map.csv",
        ),
        "inventory": write_json(
            structure_inventory(all_atoms), output_dir / "structure_inventory.json"
        ),
    }
    assembly_verified = bool(
        config.get("structure.biological_assembly_verified", False)
    )
    manifest = build_manifest(
        stage="structure",
        config_path=config.path,
        inputs=[source],
        parameters={
            "format": fmt,
            "label_asym_id": chain,
            "source_model_number": model_number,
            "assembly_id": config.get("structure.assembly_id", "unspecified"),
            "biological_assembly_verified": assembly_verified,
            "alternate_location_policy": (
                "highest occupancy; ties blank, A, lexical"
            ),
            "contact_cutoff_angstrom": config.get(
                "structure.contact_cutoff_angstrom", 8.0
            ),
            "n_residues": len(table),
            "n_context": int(table["dfi_context_mask"].sum()),
            "n_analysis": int(table["analysis_mask"].sum()),
        },
        warnings=[] if assembly_verified else [
            "Biological assembly has not been marked as verified."
        ],
    )
    outputs["manifest"] = write_json(manifest, output_dir / "manifest.structure.json")
    return outputs
