from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .dci import calculate


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}
BACKBONE = {"N", "CA", "C", "O"}
MUTATION = re.compile(r"^([A-Z])(\d+)([A-Z])$")
RCSB_DOWNLOAD = "https://files.rcsb.org/download/{filename}"
SIFTS_UNIPROT = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"


class PreparationError(ValueError):
    pass


@dataclass(frozen=True)
class Atom:
    group: str
    atom_name: str
    altloc: str
    residue_name: str
    label_asym_id: str
    auth_asym_id: str
    entity_id: str
    label_seq_id: int
    auth_seq_id: str
    insertion_code: str
    model_number: int
    x: float
    y: float
    z: float
    occupancy: float
    element: str


def _clean(value: str) -> str:
    return "" if value in {".", "?"} else value


def _float(value: str, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise PreparationError(f"Expected integer {field}, received {value!r}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(value: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    return path


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def parse_cif_loops(path: Path) -> dict[str, list[dict[str, str]]]:
    """Parse mmCIF loops without requiring a compiled structural dependency."""

    lines = path.read_text(encoding="utf-8").splitlines()
    loops: dict[str, list[dict[str, str]]] = {}
    wanted = {"_atom_site", "_entity_poly_seq", "_pdbx_struct_assembly"}
    index = 0
    while index < len(lines):
        if lines[index].strip().lower() != "loop_":
            index += 1
            continue
        index += 1
        tags: list[str] = []
        while index < len(lines) and lines[index].lstrip().startswith("_"):
            tags.append(lines[index].strip().split()[0])
            index += 1
        if not tags:
            continue
        category = tags[0].split(".", 1)[0]
        if category not in wanted:
            while index < len(lines):
                stripped = lines[index].strip()
                if (
                    stripped == "#"
                    or stripped.lower() == "loop_"
                    or stripped.lower().startswith(("data_", "save_"))
                ):
                    break
                index += 1
            continue
        values: list[str] = []
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            if (
                stripped == "#"
                or stripped.lower() == "loop_"
                or stripped.lower().startswith(("data_", "save_"))
                or (stripped.startswith("_") and len(values) % len(tags) == 0)
            ):
                break
            if stripped.startswith(";"):
                raise PreparationError(
                    f"Unsupported multiline value inside {category} loop"
                )
            try:
                values.extend(shlex.split(lines[index], posix=True, comments=False))
            except ValueError as exc:
                raise PreparationError(
                    f"Could not tokenize mmCIF line {index + 1}"
                ) from exc
            index += 1
        if len(values) % len(tags):
            raise PreparationError(
                f"Incomplete {category} loop: {len(values)} values for {len(tags)} columns"
            )
        short_tags = [tag.split(".", 1)[1] for tag in tags]
        rows = [
            dict(zip(short_tags, values[offset:offset + len(tags)]))
            for offset in range(0, len(values), len(tags))
        ]
        loops.setdefault(category, []).extend(rows)
    return loops


def atoms_from_loops(loops: dict[str, list[dict[str, str]]]) -> list[Atom]:
    rows = loops.get("_atom_site", [])
    if not rows:
        raise PreparationError("The mmCIF file does not contain an _atom_site loop")

    def get(row: dict[str, str], *names: str, default: str = "") -> str:
        for name in names:
            if name in row:
                return row[name]
        return default

    atoms: list[Atom] = []
    for row in rows:
        label_seq = _clean(get(row, "label_seq_id"))
        if not label_seq:
            continue
        atoms.append(
            Atom(
                group=get(row, "group_PDB").upper(),
                atom_name=_clean(get(row, "label_atom_id", "auth_atom_id")).upper(),
                altloc=_clean(get(row, "label_alt_id")),
                residue_name=_clean(
                    get(row, "label_comp_id", "auth_comp_id")
                ).upper(),
                label_asym_id=_clean(get(row, "label_asym_id")),
                auth_asym_id=_clean(get(row, "auth_asym_id")),
                entity_id=_clean(get(row, "label_entity_id")),
                label_seq_id=_int(label_seq, "label_seq_id"),
                auth_seq_id=_clean(get(row, "auth_seq_id", default=label_seq)),
                insertion_code=_clean(get(row, "pdbx_PDB_ins_code")),
                model_number=_int(
                    _clean(get(row, "pdbx_PDB_model_num", default="1")) or "1",
                    "pdbx_PDB_model_num",
                ),
                x=_float(get(row, "Cartn_x")),
                y=_float(get(row, "Cartn_y")),
                z=_float(get(row, "Cartn_z")),
                occupancy=_float(get(row, "occupancy", default="1"), 1.0),
                element=_clean(get(row, "type_symbol")),
            )
        )
    return atoms


def resolve_chain(
    atoms: Iterable[Atom], requested: str, model_number: int
) -> tuple[str, str, str, list[Atom]]:
    model_atoms = [atom for atom in atoms if atom.model_number == model_number]
    triples = sorted(
        {
            (atom.label_asym_id, atom.auth_asym_id, atom.entity_id)
            for atom in model_atoms
            if atom.group == "ATOM"
        }
    )
    matches = [
        triple for triple in triples
        if requested in {triple[0], triple[1]}
    ]
    if not matches:
        available = [
            {"label_asym_id": label, "auth_asym_id": auth, "entity_id": entity}
            for label, auth, entity in triples
        ]
        raise PreparationError(
            f"Chain {requested!r} was not found in model {model_number}. "
            f"Available polymer chains: {available}"
        )
    if len(matches) != 1:
        raise PreparationError(
            f"Chain {requested!r} is ambiguous after assembly/model resolution: {matches}"
        )
    label, auth, entity = matches[0]
    selected = [
        atom for atom in model_atoms
        if atom.label_asym_id == label and atom.auth_asym_id == auth
    ]
    return label, auth, entity, selected


def select_altlocs(atoms: Iterable[Atom]) -> list[Atom]:
    groups: dict[tuple[str, int, str], list[Atom]] = {}
    for atom in atoms:
        groups.setdefault(
            (atom.label_asym_id, atom.label_seq_id, atom.atom_name), []
        ).append(atom)

    def altloc_rank(value: str) -> tuple[int, str]:
        if value == "":
            return (0, "")
        if value == "A":
            return (1, "A")
        return (2, value)

    selected = [
        sorted(group, key=lambda atom: (-atom.occupancy, altloc_rank(atom.altloc)))[0]
        for group in groups.values()
    ]
    return sorted(selected, key=lambda atom: (atom.label_seq_id, atom.atom_name))


def deposited_sequence(
    loops: dict[str, list[dict[str, str]]], entity_id: str
) -> str:
    rows = [
        row for row in loops.get("_entity_poly_seq", [])
        if _clean(row.get("entity_id", "")) == entity_id
    ]
    if not rows:
        raise PreparationError(
            f"No deposited polymer sequence found for entity {entity_id}"
        )
    rows.sort(key=lambda row: _int(row["num"], "entity_poly_seq.num"))
    return "".join(AA3_TO_1.get(_clean(row["mon_id"]).upper(), "X") for row in rows)


def read_fasta(path: Path) -> str:
    sequence = "".join(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ).upper()
    if not sequence or re.search(r"[^ACDEFGHIKLMNPQRSTVWYOUX]", sequence):
        raise PreparationError(f"Invalid protein FASTA sequence: {path}")
    return sequence


def parse_mutations(value: object) -> list[tuple[str, int, str]] | None:
    parts = str(value).strip().upper().split(":")
    parsed = []
    for part in parts:
        match = MUTATION.fullmatch(part)
        if not match:
            return None
        wt, position, mutant = match.groups()
        parsed.append((wt, int(position), mutant))
    return parsed


def reconstruct_dms_sequence(
    dms: pd.DataFrame,
    *,
    mutation_column: str,
    sequence_column: str,
) -> str | None:
    if sequence_column not in dms or mutation_column not in dms:
        return None
    candidates: set[str] = set()
    for _, row in dms[[mutation_column, sequence_column]].dropna().iterrows():
        mutations = parse_mutations(row[mutation_column])
        sequence = list(str(row[sequence_column]).strip().upper())
        if not mutations or not sequence:
            continue
        valid = True
        for wt, position, mutant in mutations:
            if position < 1 or position > len(sequence) or sequence[position - 1] != mutant:
                valid = False
                break
            sequence[position - 1] = wt
        if valid:
            candidates.add("".join(sequence))
        if len(candidates) > 1:
            raise PreparationError(
                "DMS rows reconstruct more than one WT assay sequence"
            )
    return next(iter(candidates)) if candidates else None


def global_alignment(
    observed: str, target: str
) -> tuple[dict[int, tuple[int, bool]], list[dict[str, object]]]:
    n, m = len(observed), len(target)
    score = np.zeros((n + 1, m + 1), dtype=int)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)
    gap = -2
    score[:, 0] = np.arange(n + 1) * gap
    score[0, :] = np.arange(m + 1) * gap
    trace[1:, 0], trace[0, 1:] = 1, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (
                score[i - 1, j - 1] + (2 if observed[i - 1] == target[j - 1] else -1),
                score[i - 1, j] + gap,
                score[i, j - 1] + gap,
            )
            move = int(np.argmax(choices))
            score[i, j], trace[i, j] = choices[move], move
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i or j:
        move = trace[i, j]
        if i and j and move == 0:
            pairs.append((i, j))
            i, j = i - 1, j - 1
        elif i and (j == 0 or move == 1):
            pairs.append((i, None))
            i -= 1
        else:
            pairs.append((None, j))
            j -= 1
    pairs.reverse()
    mapping: dict[int, tuple[int, bool]] = {}
    rows = []
    for column, (observed_position, target_position) in enumerate(pairs, start=1):
        observed_aa = observed[observed_position - 1] if observed_position else "-"
        target_aa = target[target_position - 1] if target_position else "-"
        identity = bool(observed_position and target_position and observed_aa == target_aa)
        if observed_position and target_position:
            mapping[observed_position] = (target_position, identity)
        rows.append(
            {
                "alignment_column": column,
                "observed_ordinal": observed_position,
                "target_position": target_position,
                "observed_aa": observed_aa,
                "target_aa": target_aa,
                "identity": identity,
            }
        )
    return mapping, rows


def build_source_of_truth(
    atoms: list[Atom],
    *,
    pdb_id: str,
    label_chain: str,
    auth_chain: str,
    entity_id: str,
    target_sequence: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = select_altlocs(atoms)
    groups: dict[int, list[Atom]] = {}
    for atom in selected:
        if atom.residue_name in AA3_TO_1:
            groups.setdefault(atom.label_seq_id, []).append(atom)
    if not groups:
        raise PreparationError("No recognized amino-acid residues in the resolved chain")
    seq_ids = sorted(groups)
    observed_sequence = "".join(
        AA3_TO_1[groups[seq_id][0].residue_name] for seq_id in seq_ids
    )
    mapping, alignment_rows = global_alignment(observed_sequence, target_sequence)
    rows: list[dict[str, object]] = []
    mapped_target_positions: set[int] = set()
    for ordinal, seq_id in enumerate(seq_ids, start=1):
        residue_atoms = groups[seq_id]
        atom_map = {atom.atom_name: atom for atom in residue_atoms}
        representative = residue_atoms[0]
        target_position, identity = mapping.get(ordinal, (None, False))
        if target_position is not None:
            mapped_target_positions.add(target_position)
        ca = atom_map.get("CA")
        backbone_complete = BACKBONE.issubset(atom_map)
        rows.append(
            {
                "structure_instance_id": f"{pdb_id}_{label_chain}",
                "pdb_id": pdb_id,
                "entity_id": entity_id,
                "label_asym_id": label_chain,
                "auth_asym_id": auth_chain,
                "label_seq_id": seq_id,
                "auth_seq_id": representative.auth_seq_id,
                "insertion_code": representative.insertion_code,
                "residue_name": representative.residue_name,
                "observed_aa": AA3_TO_1[representative.residue_name],
                "target_position": target_position,
                "target_aa": (
                    target_sequence[target_position - 1] if target_position else None
                ),
                "dms_position": target_position,
                "mapping_identity_ok": identity,
                "has_n": "N" in atom_map,
                "has_ca": ca is not None,
                "has_c": "C" in atom_map,
                "has_o": "O" in atom_map,
                "ca_x": ca.x if ca else np.nan,
                "ca_y": ca.y if ca else np.nan,
                "ca_z": ca.z if ca else np.nan,
                "mean_occupancy": float(
                    np.mean([atom.occupancy for atom in residue_atoms])
                ),
                "selected_altlocs": ",".join(
                    sorted({atom.altloc or "." for atom in residue_atoms})
                ),
                "dfi_context_mask": ca is not None,
                "analysis_mask": bool(backbone_complete and identity),
                "mapping_status": (
                    "matched"
                    if identity
                    else "identity_mismatch"
                    if target_position
                    else "unmapped_structure_residue"
                ),
            }
        )
    for position, aa in enumerate(target_sequence, start=1):
        if position in mapped_target_positions:
            continue
        rows.append(
            {
                "structure_instance_id": f"{pdb_id}_{label_chain}",
                "pdb_id": pdb_id,
                "entity_id": entity_id,
                "label_asym_id": label_chain,
                "auth_asym_id": auth_chain,
                "label_seq_id": pd.NA,
                "auth_seq_id": pd.NA,
                "insertion_code": "",
                "residue_name": pd.NA,
                "observed_aa": pd.NA,
                "target_position": position,
                "target_aa": aa,
                "dms_position": position,
                "mapping_identity_ok": False,
                "has_n": False,
                "has_ca": False,
                "has_c": False,
                "has_o": False,
                "ca_x": np.nan,
                "ca_y": np.nan,
                "ca_z": np.nan,
                "mean_occupancy": np.nan,
                "selected_altlocs": "",
                "dfi_context_mask": False,
                "analysis_mask": False,
                "mapping_status": "missing_structure_residue",
            }
        )
    table = pd.DataFrame(rows)
    structure_rows = table["label_seq_id"].notna()
    completeness = table[["has_n", "has_ca", "has_c", "has_o"]].sum(axis=1) / 4
    table["coordinate_quality"] = (
        completeness * table["mean_occupancy"].fillna(0).clip(0, 1)
    )
    mapped = set(
        table.loc[table["mapping_identity_ok"], "target_position"].dropna().astype(int)
    )
    table["gap_adjacent"] = table["target_position"].apply(
        lambda value: int(
            pd.notna(value)
            and (
                (int(value) > 1 and int(value) - 1 not in mapped)
                or (
                    int(value) < len(target_sequence)
                    and int(value) + 1 not in mapped
                )
            )
        )
    )
    table["contact_number"] = np.nan
    context = structure_rows & table["dfi_context_mask"]
    coordinates = table.loc[context, ["ca_x", "ca_y", "ca_z"]].to_numpy(float)
    if len(coordinates):
        distances = np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], axis=2
        )
        table.loc[context, "contact_number"] = (
            (distances <= 8.0) & (distances > 0)
        ).sum(axis=1)
    table["dfi_serial"] = pd.NA
    context_indices = table.index[context]
    table.loc[context_indices, "dfi_serial"] = np.arange(1, len(context_indices) + 1)
    table = table.sort_values(
        ["target_position", "label_seq_id"], na_position="last"
    ).reset_index(drop=True)
    for column in ("label_seq_id", "target_position", "dms_position", "dfi_serial"):
        table[column] = table[column].astype("Int64")
    return table, pd.DataFrame(alignment_rows)


def write_pdb(
    atoms: list[Atom],
    table: pd.DataFrame,
    path: Path,
    *,
    mask: str,
    ca_only: bool,
) -> Path:
    selected = select_altlocs(atoms)
    allowed = set(table.loc[table[mask], "label_seq_id"].dropna().astype(int))
    dfi_serial = table.dropna(subset=["label_seq_id"]).set_index(
        "label_seq_id"
    )["dfi_serial"].to_dict()
    lines = []
    serial = 1
    for atom in selected:
        if atom.label_seq_id not in allowed:
            continue
        if ca_only and atom.atom_name != "CA":
            continue
        if not ca_only and atom.atom_name not in BACKBONE:
            continue
        atom_serial = int(dfi_serial[atom.label_seq_id]) if ca_only else serial
        auth_number = int(re.sub(r"[^0-9-]", "", atom.auth_seq_id) or atom.label_seq_id)
        chain = (atom.auth_asym_id or atom.label_asym_id or "A")[0]
        lines.append(
            f"ATOM  {atom_serial:5d} {atom.atom_name:^4s} {atom.residue_name:>3s} "
            f"{chain}{auth_number:4d}{(atom.insertion_code or ' '):1s}   "
            f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
            f"{atom.occupancy:6.2f}{0.0:6.2f}          {atom.element:>2s}"
        )
        serial += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + ["TER", "END"]) + "\n", encoding="utf-8")
    return path


def structure_inventory(
    all_atoms: list[Atom], selected_atoms: list[Atom], loops: dict[str, list[dict[str, str]]]
) -> dict[str, object]:
    residue_keys = {
        (
            atom.group, atom.label_asym_id, atom.auth_asym_id,
            atom.label_seq_id, atom.residue_name,
        )
        for atom in all_atoms
    }
    waters = {"HOH", "WAT", "DOD"}
    assemblies = [
        {
            "id": _clean(row.get("id", "")),
            "details": _clean(row.get("details", "")),
            "oligomeric_count": _clean(row.get("oligomeric_count", "")),
        }
        for row in loops.get("_pdbx_struct_assembly", [])
    ]
    return {
        "atom_count_all_chains": len(all_atoms),
        "atom_count_selected_chain": len(selected_atoms),
        "polymer_chain_pairs": sorted(
            {
                f"label={atom.label_asym_id};auth={atom.auth_asym_id};entity={atom.entity_id}"
                for atom in all_atoms if atom.group == "ATOM"
            }
        ),
        "assemblies_declared_in_source": assemblies,
        "alternate_location_labels": sorted(
            {atom.altloc for atom in all_atoms if atom.altloc}
        ),
        "minimum_occupancy": min(
            (atom.occupancy for atom in all_atoms), default=None
        ),
        "nonstandard_polymer_residue_names": sorted(
            {
                key[4] for key in residue_keys
                if key[0] == "ATOM" and key[4] not in AA3_TO_1
            }
        ),
        "hetero_residue_names_excluding_water": sorted(
            {
                key[4] for key in residue_keys
                if key[0] == "HETATM" and key[4] not in waters
            }
        ),
        "water_residue_count": sum(key[4] in waters for key in residue_keys),
    }


def retrieve(url: str, destination: Path) -> Path:
    request = urllib.request.Request(
        url, headers={"User-Agent": "ResiDYN/0.2 (+https://github.com/DavidInabinette/ResiDYN)"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise PreparationError(f"Download failed ({exc.code}): {url}") from exc
    except urllib.error.URLError as exc:
        raise PreparationError(f"Could not reach {url}: {exc.reason}") from exc
    if not payload:
        raise PreparationError(f"Downloaded file is empty: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def fetch_sifts(pdb_id: str) -> tuple[dict[str, object] | None, str | None]:
    request = urllib.request.Request(
        SIFTS_UNIPROT.format(pdb_id=pdb_id.lower()),
        headers={"User-Agent": "ResiDYN/0.2"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8")), None
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        return None, f"SIFTS retrieval failed: {exc}"


def sifts_accessions(
    payload: dict[str, object] | None, pdb_id: str, label_chain: str, auth_chain: str
) -> list[str]:
    if not payload:
        return []
    entry = payload.get(pdb_id.lower(), {})
    uniprot = entry.get("UniProt", {}) if isinstance(entry, dict) else {}
    accessions = []
    if isinstance(uniprot, dict):
        for accession, record in uniprot.items():
            mappings = record.get("mappings", []) if isinstance(record, dict) else []
            if any(
                mapping.get("chain_id") == auth_chain
                or mapping.get("struct_asym_id") == label_chain
                for mapping in mappings
                if isinstance(mapping, dict)
            ):
                accessions.append(accession)
    return sorted(accessions)


def make_variants(
    dms: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    mutation_column: str,
    score_column: str,
    assay_id: str,
) -> pd.DataFrame:
    if mutation_column not in dms or score_column not in dms:
        raise PreparationError(
            f"DMS CSV requires {mutation_column!r} and {score_column!r} columns"
        )
    parsed = dms[mutation_column].map(parse_mutations)
    output = pd.DataFrame(
        {
            "assay_id": assay_id,
            "source_row": np.arange(len(dms)),
            "mutation": dms[mutation_column].astype(str),
            "dms_score_raw": pd.to_numeric(dms[score_column], errors="coerce"),
            "parsed": parsed,
        }
    )
    output["is_single_substitution"] = parsed.map(
        lambda value: bool(value and len(value) == 1)
    )
    output["wt"] = parsed.map(
        lambda value: value[0][0] if value and len(value) == 1 else pd.NA
    )
    output["target_position"] = parsed.map(
        lambda value: value[0][1] if value and len(value) == 1 else pd.NA
    ).astype("Int64")
    output["mutant"] = parsed.map(
        lambda value: value[0][2] if value and len(value) == 1 else pd.NA
    )
    crosswalk = truth.loc[
        truth["target_position"].notna(),
        [
            "target_position", "structure_instance_id", "label_asym_id",
            "label_seq_id", "target_aa", "analysis_mask",
        ],
    ].drop_duplicates("target_position")
    output = output.merge(crosswalk, on="target_position", how="left", validate="many_to_one")
    output["wt_identity_ok"] = (
        output["wt"].astype("string")
        .eq(output["target_aa"].astype("string"))
        .fillna(False)
    )
    analysis_mask = output["analysis_mask"].astype("boolean").fillna(False)
    output["exclusion_reason"] = np.select(
        [
            ~output["is_single_substitution"],
            output["dms_score_raw"].isna(),
            output["structure_instance_id"].isna(),
            ~output["wt_identity_ok"],
            ~analysis_mask,
        ],
        [
            "not_single_substitution",
            "missing_or_non_numeric_score",
            "unmapped_target_position",
            "wild_type_identity_mismatch",
            "outside_analysis_mask",
        ],
        default="",
    )
    output["eligible"] = output["exclusion_reason"].eq("")
    output["dms_score_z"] = np.nan
    scores = output.loc[output["eligible"], "dms_score_raw"].astype(float)
    if len(scores) > 1 and float(scores.std(ddof=0)) > 0:
        output.loc[output["eligible"], "dms_score_z"] = (
            scores - float(scores.mean())
        ) / float(scores.std(ddof=0))
    output = output.drop(columns=["parsed"])
    return output


def _copy_or_download_source(args: object, source_dir: Path) -> tuple[Path, str]:
    pdb_id = args.pdb_id.lower()
    filename = (
        f"{pdb_id}-assembly{args.assembly}.cif"
        if args.assembly else f"{pdb_id}.cif"
    )
    destination = source_dir / filename
    if args.cif:
        local = args.cif.expanduser().resolve()
        if not local.is_file():
            raise PreparationError(f"Local mmCIF does not exist: {local}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, destination)
        return destination, f"local:{local}"
    url = RCSB_DOWNLOAD.format(filename=filename)
    retrieve(url, destination)
    return destination, url


def run_prepare(args: object) -> dict[str, object]:
    pdb_id = str(args.pdb_id).upper()
    if not re.fullmatch(r"[A-Za-z0-9]{4}", pdb_id):
        raise PreparationError(f"Invalid four-character PDB ID: {pdb_id}")
    chain_slug = re.sub(r"[^A-Za-z0-9_.-]", "_", str(args.chain))
    target_dir = args.output.expanduser().resolve() / f"{pdb_id}_{chain_slug}"
    source_dir = target_dir / "source"
    tables_dir = target_dir / "tables"
    structures_dir = target_dir / "structures"
    for directory in (source_dir, tables_dir, structures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    request_record = {
        "pdb_id": pdb_id,
        "requested_chain": args.chain,
        "assembly": args.assembly,
        "model": args.model,
        "dms": str(args.dms.resolve()) if args.dms else None,
        "target_fasta": str(args.target_fasta.resolve()) if args.target_fasta else None,
        "features": str(args.features.resolve()) if args.features else None,
        "assay_id": args.assay_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(request_record, target_dir / "request.json")
    source_path, source_origin = _copy_or_download_source(args, source_dir)
    loops = parse_cif_loops(source_path)
    all_atoms = atoms_from_loops(loops)
    label_chain, auth_chain, entity_id, chain_atoms = resolve_chain(
        all_atoms, str(args.chain), int(args.model)
    )
    deposited = deposited_sequence(loops, entity_id)

    warnings: list[str] = []
    review: list[str] = []
    if args.assembly is None:
        review.append(
            "Biological assembly has not been explicitly selected; coordinates "
            "come from the deposited asymmetric unit."
        )

    dms = pd.read_csv(args.dms) if args.dms else None
    if args.target_fasta:
        target_sequence = read_fasta(args.target_fasta)
        target_source = "user_verified_fasta"
        if dms is None:
            review.append(
                "No DMS table was supplied. The target sequence is verified, but "
                "the source of truth is not yet validation-ready."
            )
    elif dms is not None:
        reconstructed = reconstruct_dms_sequence(
            dms,
            mutation_column=args.mutation_column,
            sequence_column=args.sequence_column,
        )
        if reconstructed:
            target_sequence = reconstructed
            target_source = "reconstructed_from_dms_mutated_sequences"
        else:
            target_sequence = deposited
            target_source = "deposited_polymer_sequence_provisional"
            review.append(
                "The DMS WT sequence could not be reconstructed. The deposited "
                "polymer sequence was used provisionally and requires confirmation."
            )
    else:
        target_sequence = deposited
        target_source = "deposited_polymer_sequence_provisional"
        review.append(
            "No DMS assay or verified target sequence was supplied. This is a "
            "provisional structural source of truth."
        )

    truth, alignment = build_source_of_truth(
        chain_atoms,
        pdb_id=pdb_id,
        label_chain=label_chain,
        auth_chain=auth_chain,
        entity_id=entity_id,
        target_sequence=target_sequence,
    )
    mismatches = truth["mapping_status"].eq("identity_mismatch")
    if mismatches.any():
        review.append(
            f"{int(mismatches.sum())} aligned structure/DMS residue identities differ."
        )

    write_pdb(
        chain_atoms, truth, structures_dir / "dfi_context.pdb",
        mask="dfi_context_mask", ca_only=True,
    )
    write_pdb(
        chain_atoms, truth, structures_dir / "mpnn_backbone.pdb",
        mask="analysis_mask", ca_only=False,
    )

    dfi_qc: dict[str, object] | None = None
    if not args.skip_dfi:
        context = truth.loc[truth["dfi_context_mask"]].sort_values("dfi_serial")
        dfi, response, dfi_qc = calculate(
            context[["ca_x", "ca_y", "ca_z"]].to_numpy(float),
            direction_count=int(args.directions),
            rotation_trials=int(args.rotation_trials),
        )
        dfi_values = context[["dfi_serial"]].copy()
        for column in dfi:
            dfi_values[column] = dfi[column].to_numpy()
        truth = truth.merge(dfi_values, on="dfi_serial", how="left", validate="many_to_one")
        np.save(target_dir / "perturbation_matrix.npy", response)
        if not dfi_qc["passed"]:
            review.append("DFI spectral or rotation QC did not pass production thresholds.")

    if args.features:
        features_path = args.features.expanduser().resolve()
        if not features_path.is_file():
            raise PreparationError(f"Residue feature CSV does not exist: {features_path}")
        features = pd.read_csv(features_path)
        if "target_position" not in features:
            raise PreparationError(
                "Residue feature CSV requires a target_position column"
            )
        if features["target_position"].duplicated().any():
            raise PreparationError(
                "Residue feature CSV contains duplicate target_position values"
            )
        if "wt" in features:
            identity = truth[["target_position", "target_aa"]].drop_duplicates(
                "target_position"
            ).merge(
                features[["target_position", "wt"]],
                on="target_position",
                how="inner",
                validate="one_to_one",
            )
            mismatch = identity["target_aa"].astype(str).str.upper().ne(
                identity["wt"].astype(str).str.upper()
            )
            if mismatch.any():
                positions = identity.loc[mismatch, "target_position"].head(20).tolist()
                raise PreparationError(
                    f"Residue feature WT mismatch at positions {positions}"
                )
            features = features.drop(columns=["wt"])
        overlap = (set(features) & set(truth)) - {"target_position"}
        if overlap:
            raise PreparationError(
                f"Residue feature columns would overwrite source-of-truth fields: "
                f"{sorted(overlap)}"
            )
        truth = truth.merge(
            features, on="target_position", how="left", validate="many_to_one"
        )

    sifts_payload = None
    if not args.skip_sifts:
        sifts_payload, sifts_warning = fetch_sifts(pdb_id)
        if sifts_warning:
            warnings.append(sifts_warning)
        elif sifts_payload is not None:
            write_json(sifts_payload, target_dir / "sifts_uniprot.json")
    accessions = sifts_accessions(
        sifts_payload, pdb_id, label_chain, auth_chain
    )

    truth.to_csv(tables_dir / "source_of_truth.csv", index=False)
    alignment.to_csv(tables_dir / "alignment.csv", index=False)
    exclusions = truth.loc[
        ~truth["analysis_mask"],
        [
            "target_position", "label_seq_id", "observed_aa", "target_aa",
            "mapping_status", "has_n", "has_ca", "has_c", "has_o",
        ],
    ]
    exclusions.to_csv(tables_dir / "exclusions.csv", index=False)

    variants_path = None
    if dms is not None:
        assay_id = args.assay_id or args.dms.stem
        variants = make_variants(
            dms,
            truth,
            mutation_column=args.mutation_column,
            score_column=args.score_column,
            assay_id=assay_id,
        )
        variants_path = tables_dir / "variants.csv"
        variants.to_csv(variants_path, index=False)
        if not variants["eligible"].any():
            review.append("No DMS variants are currently eligible for validation.")

    inventory = structure_inventory(all_atoms, chain_atoms, loops)
    write_json(inventory, target_dir / "structure_inventory.json")
    sequences = (
        f">deposited|{pdb_id}|entity={entity_id}\n{deposited}\n"
        f">target|source={target_source}\n{target_sequence}\n"
    )
    (target_dir / "sequences.fasta").write_text(sequences, encoding="utf-8")

    status = "frozen" if not review else "needs_review"
    qc = {
        "status": status,
        "review_items": review,
        "warnings": warnings,
        "mapping": {
            "target_length": len(target_sequence),
            "observed_structure_residues": int(truth["label_seq_id"].notna().sum()),
            "identity_matched_residues": int(truth["mapping_identity_ok"].sum()),
            "dfi_context_residues": int(truth["dfi_context_mask"].sum()),
            "analysis_residues": int(truth["analysis_mask"].sum()),
            "identity_mismatches": int(mismatches.sum()),
        },
        "dfi": dfi_qc,
    }
    write_json(qc, target_dir / "phase1_qc.json")
    manifest = {
        "status": status,
        "pdb_id": pdb_id,
        "source": {
            "origin": source_origin,
            "local_file": str(source_path),
            "sha256": sha256(source_path),
        },
        "resolved_structure": {
            "requested_chain": args.chain,
            "label_asym_id": label_chain,
            "auth_asym_id": auth_chain,
            "entity_id": entity_id,
            "model_number": args.model,
            "assembly": args.assembly or "deposited_asymmetric_unit",
        },
        "sequence": {
            "target_source": target_source,
            "deposited_length": len(deposited),
            "target_length": len(target_sequence),
        },
        "external_mappings": {"uniprot_accessions": accessions},
        "method": {
            "alternate_location_policy": (
                "highest occupancy; ties prefer blank, A, then lexical"
            ),
            "dfi_spring_law": "gamma^3/d^6",
            "dfi_directions": args.directions,
            "rotation_trials": args.rotation_trials,
        },
        "canonical_table": str(tables_dir / "source_of_truth.csv"),
    }
    write_json(manifest, target_dir / "resolved_manifest.json")

    report_lines = [
        f"# Phase 1 report: {pdb_id} chain {args.chain}",
        "",
        f"**Status:** `{status}`",
        "",
        "## Resolved structure",
        "",
        f"- Label chain: `{label_chain}`",
        f"- Author chain: `{auth_chain}`",
        f"- Polymer entity: `{entity_id}`",
        f"- Model: `{args.model}`",
        f"- Assembly: `{args.assembly or 'deposited asymmetric unit'}`",
        f"- UniProt candidates from SIFTS: {', '.join(accessions) or 'none resolved'}",
        "",
        "## Mapping summary",
        "",
        f"- Target sequence source: `{target_source}`",
        f"- Target residues: {len(target_sequence)}",
        f"- Observed structure residues: {int(truth['label_seq_id'].notna().sum())}",
        f"- Identity-matched residues: {int(truth['mapping_identity_ok'].sum())}",
        f"- DFI-context residues: {int(truth['dfi_context_mask'].sum())}",
        f"- Analysis residues: {int(truth['analysis_mask'].sum())}",
        "",
        "## Items requiring review",
        "",
        *([f"- {item}" for item in review] if review else ["- None"]),
        "",
        "## Warnings",
        "",
        *([f"- {item}" for item in warnings] if warnings else ["- None"]),
        "",
        "The canonical Phase 1 table is `tables/source_of_truth.csv`.",
    ]
    (target_dir / "phase1_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return {
        "status": status,
        "target_directory": target_dir,
        "source_of_truth": tables_dir / "source_of_truth.csv",
        "phase1_report": target_dir / "phase1_report.md",
        "variants": variants_path,
        "review_items": review,
        "warnings": warnings,
    }
