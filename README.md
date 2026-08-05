# ResiDYN

**Residue dynamics analysis of ProteinMPNN–DMS residuals**

ResiDYN tests whether wild-type residue flexibility helps explain where
ProteinMPNN variant-effect predictions disagree with deep mutational scanning
measurements after structural, evolutionary, and substitution-level controls.

## The simplified Phase 1 workflow

The smallest valid structural run requires only a PDB ID and chain:

```bash
python resiDYN.py prepare 1BTL A
```

ResiDYN then:

1. downloads the original PDBx/mmCIF file from RCSB PDB;
2. resolves the supplied author or label chain;
3. identifies the polymer entity and deposited sequence;
4. selects coordinates deterministically;
5. inventories missing atoms, alternate locations, ligands, waters, metals,
   and nonstandard residues;
6. aligns the observed structure to the deposited sequence;
7. builds separate DFI-context and analysis masks;
8. calculates DFI and its spectral and rotation QC;
9. writes the Phase 1 source-of-truth table and review report.

This produces a **provisional structural source of truth**. A PDB chain does not
uniquely identify a DMS construct, so a DMS file is needed before the mapping
can be considered validation-ready:

```bash
python resiDYN.py prepare 1BTL A \
  --dms data/BLAT_ECOLX_Stiffler_2015.csv
```

ProteinGym files normally include `mutant`, `mutated_sequence`, and
`DMS_score`. ResiDYN reconstructs the WT assay sequence from those rows and
requires all reconstructed sequences to agree.

You can also provide an explicit sequence:

```bash
python resiDYN.py prepare 1BTL A \
  --dms data/tem1.csv \
  --target-fasta data/tem1_wt.fasta
```

Optional residue-level features such as ConSurf, SASA, or secondary structure
can be supplied in one CSV keyed by `target_position`:

```bash
python resiDYN.py prepare 1BTL A \
  --dms data/tem1.csv \
  --features data/tem1_residue_features.csv
```

If the feature file includes a `wt` column, ResiDYN validates its residue
identities before merging it into the source of truth.

If ResiDYN cannot make a scientifically safe decision, the run is marked
`needs_review`. It does not silently repair structures or choose among
ambiguous mappings.

## Installation

ResiDYN requires Python 3.10 or newer.

```bash
git clone https://github.com/DavidInabinette/ResiDYN.git
cd ResiDYN

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Directory layout

```text
ResiDYN/
├── README.md
├── requirements.txt
├── resiDYN.py
├── src/
│   ├── prepare.py
│   ├── dci.py
│   ├── mpnn.py
│   ├── validate.py
│   └── selftest.py
└── Output/
    └── tables/
```

Target-specific folders are created automatically under `Output/`:

```text
Output/1BTL_A/
├── source/
│   └── 1btl.cif
├── structures/
│   ├── dfi_context.pdb
│   └── mpnn_backbone.pdb
├── tables/
│   ├── source_of_truth.csv
│   ├── alignment.csv
│   ├── exclusions.csv
│   └── variants.csv
├── request.json
├── resolved_manifest.json
├── structure_inventory.json
├── phase1_qc.json
└── phase1_report.md
```

`source_of_truth.csv` is the canonical Phase 1 output. Later stages should join
through its stable structure and target identifiers rather than recreating
their own residue numbering.

## Offline or local structure input

To use an existing mmCIF instead of downloading it:

```bash
python resiDYN.py prepare 1BTL A --cif path/to/1btl.cif
```

The supplied file is copied unchanged into the target's `source/` directory
and hashed.

## Biological assemblies

The default run uses the deposited asymmetric unit and records that the
biological assembly still requires review. To request a specific RCSB
biological assembly:

```bash
python resiDYN.py prepare 1BTL A --assembly 1
```

Assembly-expanded chain identifiers can differ from deposited chain
identifiers. ResiDYN stops if the requested chain is no longer unique.

## ProteinMPNN scoring

ProteinMPNN is run with its upstream repository. ResiDYN imports the conditional
probability NPZ and verifies it against the frozen target sequence:

```bash
python resiDYN.py mpnn Output/1BTL_A \
  --npz path/to/conditional_probs_only.npz \
  --commit PINNED_COMMIT \
  --checkpoint v_48_020 \
  --command-file path/to/proteinmpnn_command.txt
```

The score for mutation \(WT_i \rightarrow a\) is:

\[
\Delta\log P_{i,a}
=
\log P(a \mid S_{\setminus i},X)
-
\log P(WT_i \mid S_{\setminus i},X)
\]

## Validation

Once `variants.csv` contains ProteinMPNN scores and the source-of-truth table
contains the required covariates:

```bash
python resiDYN.py validate Output/1BTL_A
```

Validation uses paired M0/M1 ridge models with:

- residue-grouped outer folds;
- residue-grouped inner tuning;
- preprocessing fit only on training data;
- held-out Spearman correlation and standardized MAE;
- paired residue-level bootstrap uncertainty.

M0 contains ProteinMPNN and available prespecified controls. M1 adds pctDFI.
All mutations at one residue remain in the same fold.

## Self-test

Run the dependency-free structural and DFI checks:

```bash
python resiDYN.py self-test
```

Run the complete synthetic smoke test:

```bash
python resiDYN.py demo
```

The synthetic result is not biologically meaningful. It verifies parsing,
alignment, masking, DFI/DCI, ProteinMPNN import, grouped validation, QC, and
artifact generation.

## Scientific boundaries

ResiDYN tests whether flexibility predicts where ProteinMPNN errors occur and
whether adding DFI improves held-out DMS prediction. Phase 1 alone cannot prove
that flexibility causes those errors.

DFI is constant across mutations at one residue. It can explain systematic
position-level error but cannot independently explain differences between two
substitutions at the same position.

## Data sources

- [RCSB PDB file download services](https://www.rcsb.org/docs/programmatic-access/file-download-services)
- [PDBe SIFTS mappings](https://www.ebi.ac.uk/pdbe/api/doc/sifts.html)
- [ProteinGym](https://github.com/OATML-Markslab/ProteinGym)
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- [ConSurf](https://consurf.tau.ac.il/)

## Citation

ResiDYN is under active development. Until a formal release is archived, cite
the repository and exact commit used:

```text
Inabinette, D. ResiDYN: Residue dynamics analysis of ProteinMPNN–DMS residuals.
https://github.com/DavidInabinette/ResiDYN
```

## License

A project license has not yet been selected.
