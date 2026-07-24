# ProteinMPNN–DFI MVP+

This package implements the first executable Phase 1 slice of the MVP+ design:

> Does WT residue flexibility improve held-out prediction of DMS effects beyond
> ProteinMPNN and a small set of structural and evolutionary controls?

It deliberately separates data plumbing from external scientific computations.
ProteinMPNN conditional probabilities, ConSurf scores, SASA, and secondary
structure can be generated with their pinned upstream tools and imported here.
The package itself owns mapping, validation, DFI/DCI, feature assembly, grouped
cross-validation, quality control, and provenance.

## What is implemented

- deterministic PDB/mmCIF atom parsing and alternate-location selection;
- sequence alignment to a target construct with amino-acid validation;
- separate DFI-context and analysis masks;
- minimal PDB inputs and a stable DFI serial-number crosswalk;
- corrected fully connected anisotropic ENM using the legacy `gamma^3 / d^6`
  spring law;
- configurable perturbation directions, spectral QC, rotation QC, and
  functional-set-to-residue DCI;
- ProteinGym-style single-substitution ingestion;
- ProteinMPNN conditional-log-probability import and mutation scoring;
- ConSurf score import with explicit missingness/reliability;
- assembly of residue and variant tables without silently changing the
  analysis mask;
- nested residue-grouped ridge regression for paired M0/M1 out-of-fold
  predictions, metrics, and residue bootstrap uncertainty;
- source hashes and run manifests.

The package does **not** download data or guess unresolved choices. Pin the
structure, DMS assay, ProteinMPNN commit/checkpoint, ConSurf run, and feature
sources in the target YAML before interpreting results.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

No compiled structural-biology dependency is required. The mmCIF reader is
purposefully limited to `_atom_site`; the unmodified source mmCIF remains the
record of truth.

## TEM-1 first run

Copy and edit the example:

```bash
cp configs/targets/tem1.example.yaml configs/targets/tem1.yaml
```

Then run the stages:

```bash
mpnn-dfi structure configs/targets/tem1.yaml
mpnn-dfi dfi configs/targets/tem1.yaml
mpnn-dfi dms configs/targets/tem1.yaml
mpnn-dfi mpnn configs/targets/tem1.yaml
mpnn-dfi consurf configs/targets/tem1.yaml
mpnn-dfi assemble configs/targets/tem1.yaml
mpnn-dfi validate configs/targets/tem1.yaml
```

`mpnn-dfi structure` produces `residues.csv`, `dfi_input.pdb`,
`proteinmpnn_input.pdb`, and `dfi_serial_map.csv`. Run ProteinMPNN externally
against `proteinmpnn_input.pdb` using the exact command recorded in the YAML,
then point `proteinmpnn.conditional_npz` to its `.npz` output.

Use `mpnn-dfi all CONFIG` only after all external input files exist.

## Frozen conventions

- Canonical key: `(structure_instance_id, label_asym_id, label_seq_id)`.
- A residue enters the DFI context with one selected observed Cα.
- A residue enters analysis only with selected observed N, Cα, C, and O and an
  unambiguous, identity-matched target-sequence mapping.
- Alternate locations: highest occupancy wins; ties prefer blank, then `A`,
  then lexical order.
- ProteinMPNN score:
  `log P(mutant | WT context, X) - log P(WT | WT context, X)`.
- DFI matrix orientation: rows are responding residues; columns are perturbed
  residues.
- Functional DCI direction: functional set → responding residue.
- Every mutation at one residue stays in the same outer and inner CV fold.

## DFI production decision

The default config uses 256 deterministic Fibonacci-sphere directions and
checks rotation stability. The historical seven Cartesian directions remain
available as `direction_mode: legacy7` only for regression/plumbing comparison.
The spring law defaults to the supplied-code behavior, `d^-6`. Both choices are
written to the manifest and must be approved before a production analysis.

## External feature contract

The primary baseline expects the following after assembly:

- numeric: `mpnn_delta_logp`, `sasa`, `contact_number`,
  `consurf_score`, `coordinate_quality`, `gap_adjacent`;
- categorical: `secondary_structure`, `wt`, `mutant`;
- optional numeric: `ligand_distance`.

`contact_number`, coordinate completeness, and gap adjacency are generated
here. Provide SASA and secondary structure in a residue-level CSV keyed by
`target_position`. ConSurf is imported separately. Missing numeric values are
median-imputed and categorical values are most-frequent-imputed **inside each
training fold**, with missingness preserved where applicable.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic structures and do not require ProteinMPNN, ConSurf, or
network access.

