# ResiDYN

**Residue dynamics analysis of ProteinMPNN=E2=80=93DMS residuals**

ResiDYN is a validation pipeline for testing whether wild-type residue
dynamics explain systematic errors in ProteinMPNN variant-effect
predictions after controlling for structural, evolutionary, and
substitution-level features.

## North-star question

> Does wild-type residue flexibility improve held-out prediction of deep
mutational scanning effects beyond ProteinMPNN and a small set of
structural and evolutionary controls?

ResiDYN is designed around the incremental value of dynamics. It compares a
baseline model containing ProteinMPNN and prespecified control variables
with an otherwise identical model that also includes dynamic flexibility
index, or DFI.

Because DFI is constant across every mutation at a residue, it can help
explain why ProteinMPNN performs differently at one position versus
another. DFI alone cannot explain why two substitutions at the same
position have different prediction errors.

This is a predictive association test. It does not assume that dynamics
causally produces ProteinMPNN errors.

## Project status

ResiDYN is currently in Phase 1 MVP+ development.

The initial implementation supports single-target analyses and is being
developed as a TEM-1-first vertical slice. The proposed pilot cohort
includes:

- TEM-1 =CE=B2-lactamase
- dihydrofolate reductase
- ubiquitin
- GB1
- PSD95 PDZ3

The five-protein cohort is intended as a controlled pilot, not a general
protein-wide benchmark.

## Primary analysis

For each single-amino-acid substitution, ResiDYN calculates the ProteinMPNN
mutation score:

\[
\Delta \log P_{i,a}
=3D
\log P(a_i \mid S_{\setminus i}, X)
-
\log P(WT_i \mid S_{\setminus i}, X)
\]

The pipeline then compares two regularized regression models.

### Baseline model

The baseline model, \(M_0\), contains:

- ProteinMPNN \(\Delta \log P\)
- solvent-accessible surface area or burial
- C=CE=B1 contact number or packing
- secondary-structure class
- continuous conservation score
- wild-type and mutant amino-acid identities
- coordinate-quality and gap-adjacency flags
- ligand distance when relevant

### Dynamics model

The dynamics model adds residue-level flexibility:

\[
M_1 =3D M_0 + \mathrm{pctDFI}
\]

The primary result is the change in held-out Spearman correlation:

\[
\Delta \rho
=3D
\rho(M_1, DMS)
-
\rho(M_0, DMS)
\]

Positive \(\Delta \rho\) indicates that DFI adds predictive information
beyond ProteinMPNN and the prespecified controls.

## Validation design

Mutations are the observed rows, but DFI is a residue-level feature.
ResiDYN therefore keeps every mutation at the same residue in the same
cross-validation fold.

The validation procedure uses:

- residue-grouped outer cross-validation
- residue-grouped inner validation for ridge regularization
- identical folds and preprocessing for \(M_0\) and \(M_1\)
- out-of-fold predictions for every eligible mutation
- held-out Spearman correlation as the primary metric
- held-out standardized MAE as a secondary metric
- paired residue-level bootstrap uncertainty

Numeric imputation, scaling, categorical imputation, and encoding are fit
only on the training portion of each fold.

## Implemented components

The current codebase includes:

- deterministic PDB and mmCIF atom parsing
- target-sequence alignment and residue crosswalk generation
- amino-acid identity validation for data joins
- separate DFI-context and analysis coordinate masks
- minimal method-specific PDB generation
- stable DFI serial-number mapping
- corrected \(d^{-6}\) anisotropic-network DFI
- configurable perturbation-direction sampling
- Hessian spectral and pseudoinverse QC
- rigid-rotation stability testing
- functional-set-to-residue DCI
- ProteinGym-style DMS ingestion
- ProteinMPNN conditional-probability import
- ProteinMPNN mutation-score calculation
- ConSurf score and reliability import
- external structural-covariate import
- residue and variant analysis tables
- nested residue-grouped ridge validation
- paired residue bootstrap uncertainty
- stage-specific provenance manifests

## DFI implementation

The default DFI implementation follows the effective spring law found in
the supplied legacy code:

\[
k_{ij} =3D \frac{\gamma^3}{d_{ij}^6}
\]

The corrected implementation addresses two important legacy behaviors:

1. A positional argument could pass the verbosity flag into the Hessian
cutoff parameter.
2. The perturbation matrix was divided by the literal value seven rather
than the number of supplied directions.

ResiDYN uses keyword-only method settings and divides by the actual number
of perturbation directions.

### Perturbation directions

Two modes are available:

- `fibonacci`: deterministic sampling over the unit sphere
- `legacy7`: the seven historical Cartesian directions

The default MVP+ configuration uses 256 Fibonacci-sphere directions. The
seven-direction mode is retained for plumbing and regression comparisons.

### Spectral QC

Before DFI values are accepted, the pipeline evaluates:

- the expected six rigid-body near-zero modes
- significant negative Hessian modes
- the inversion threshold
- the seventh-smallest singular value
- pseudoinverse symmetry
- DFI rank stability after rigid rotations
- DFI quartile stability after rigid rotations

Production runs fail closed when the configured spectral or rotation
thresholds are not met.

## DCI orientation

The perturbation matrix is oriented as follows:

- rows represent responding residues
- columns represent perturbed residues

Functional DCI is therefore calculated in the direction:

> functional residue set =E2=86=92 responding residue

Functional DCI is currently treated as a secondary mechanistic analysis
rather than a second primary predictor across every protein.

## Coordinate masks

ResiDYN uses two separate coordinate masks.

### DFI-context mask

Includes every valid, unique, experimentally observed C=CE=B1 used to const=
ruct
the elastic network model.

### Analysis mask

Includes residues that:

- contain observed N, C=CE=B1, C, and O atoms
- map unambiguously to the DMS target sequence
- pass wild-type amino-acid identity validation

DFI is calculated using the full context mask. Statistical comparisons use
the stricter analysis mask.

Missing backbone atoms are not imputed in the primary analysis.

## Required inputs

A target run requires:

1. a verified experimental PDB or mmCIF structure
2. the verified DMS wild-type sequence
3. a ProteinGym-style single-substitution DMS table
4. ProteinMPNN conditional log probabilities
5. ConSurf scores
6. residue-level structural covariates such as SASA and secondary structure
7. a completed target configuration file

ProteinMPNN and ConSurf are run through their respective upstream tools.
ResiDYN imports and validates their outputs rather than reimplementing them=
.

## Installation

Clone the repository:

```bash
git clone https://github.com/DavidInabinette/ResiDYN.git
cd ResiDYN
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install ResiDYN:

```bash
pip install -e .
```

To install the development dependencies:

```bash
pip install -e ".[dev]"
```

ResiDYN requires Python 3.10 or newer.

## Repository structure

```text
ResiDYN/
=E2=94=9C=E2=94=80=E2=94=80 configs/
=E2=94=82   =E2=94=94=E2=94=80=E2=94=80 targets/
=E2=94=82       =E2=94=94=E2=94=80=E2=94=80 tem1.example.yaml
=E2=94=9C=E2=94=80=E2=94=80 examples/
=E2=94=82   =E2=94=94=E2=94=80=E2=94=80 make_synthetic_demo.py
=E2=94=9C=E2=94=80=E2=94=80 src/
=E2=94=82   =E2=94=94=E2=94=80=E2=94=80 mpnn_dfi/
=E2=94=82       =E2=94=9C=E2=94=80=E2=94=80 cli.py
=E2=94=82       =E2=94=9C=E2=94=80=E2=94=80 config.py
=E2=94=82       =E2=94=9C=E2=94=80=E2=94=80 conservation.py
=E2=94=82       =E2=94=9C=E2=94=80=E2=94=80 dfi.py
=E2=94=82       =E2=94=9C=E2=94=80=E2=94=80 dms.py
=E2=94=82       =E2=94=9C=E2=94=80=E2=94=80 features.py
=E2=94=82       =E2=94=9C=E2=94=80=E2=94=80 io_utils.py
=E2=94=82       =E2=94=9C=E2=94=80=E2=94=80 proteinmpnn.py
=E2=94=82       =E2=94=9C=E2=94=80=E2=94=80 structure.py
=E2=94=82       =E2=94=94=E2=94=80=E2=94=80 validation.py
=E2=94=9C=E2=94=80=E2=94=80 tests/
=E2=94=82   =E2=94=94=E2=94=80=E2=94=80 test_pipeline.py
=E2=94=9C=E2=94=80=E2=94=80 vendor/
=E2=94=82   =E2=94=94=E2=94=80=E2=94=80 README.md
=E2=94=9C=E2=94=80=E2=94=80 pyproject.toml
=E2=94=94=E2=94=80=E2=94=80 README.md
```

## Configuring a target

Start by copying the TEM-1 example configuration:

```bash
cp configs/targets/tem1.example.yaml configs/targets/tem1.yaml
```

Before running the pipeline, replace every unresolved value in the
configuration, including:

- the verified DMS wild-type sequence
- structure and assay paths
- biological-assembly verification
- the ProteinMPNN repository commit
- the ProteinMPNN checkpoint
- the exact ProteinMPNN command
- the ProteinMPNN training-exposure audit
- ConSurf output
- SASA and secondary-structure inputs
- functional residue definitions, when applicable

The example configuration contains explicit placeholders so unresolved
scientific choices cannot be mistaken for finalized settings.

## Running the pipeline

Run each stage independently:

```bash
mpnn-dfi structure configs/targets/tem1.yaml
mpnn-dfi dfi configs/targets/tem1.yaml
mpnn-dfi dms configs/targets/tem1.yaml
mpnn-dfi mpnn configs/targets/tem1.yaml
mpnn-dfi consurf configs/targets/tem1.yaml
mpnn-dfi assemble configs/targets/tem1.yaml
mpnn-dfi validate configs/targets/tem1.yaml
```

Once all external inputs are available, the complete sequence can be run
with:

```bash
mpnn-dfi all configs/targets/tem1.yaml
```

## Pipeline stages

### `structure`

Produces:

- the canonical residue table
- DFI-context and analysis masks
- a minimal C=CE=B1-only DFI structure
- a backbone-only ProteinMPNN structure
- the DFI serial-number crosswalk
- a structure inventory
- a structure-stage manifest

### `dfi`

Produces:

- raw DFI
- relative DFI
- percentile-ranked DFI
- standardized DFI
- optional functional DCI
- the normalized perturbation matrix
- spectral and rotation QC
- a DFI-stage manifest

### `dms`

Produces:

- parsed single substitutions
- standardized assay scores
- structure and sequence mappings
- eligibility flags
- explicit exclusion reasons

### `mpnn`

Imports ProteinMPNN conditional probabilities and adds:

- wild-type log probability
- mutant log probability
- mutation \(\Delta \log P\)
- checkpoint and command provenance

### `consurf`

Imports:

- continuous ConSurf scores
- ConSurf grades
- reliability flags
- missingness information

### `assemble`

Produces the two primary analysis products:

- `table_a_residues.csv`
- `table_b_variants.csv`

### `validate`

Produces:

- paired M0 and M1 out-of-fold predictions
- fold assignments
- selected ridge penalties
- held-out metrics
- residue-bootstrap results
- the final validation summary

## Output products

### Table A

One row per structurally mapped residue, including:

- canonical structure identifiers
- author and target numbering
- coordinate masks and quality fields
- DFI and DFI QC fields
- structural covariates
- conservation and reliability
- optional ligand distance
- exclusion and missingness flags

### Table B

One row per DMS mutation, including:

- target and assay identifiers
- canonical residue key
- wild-type and mutant amino acids
- raw and standardized DMS scores
- ProteinMPNN mutation score
- residue-level features
- eligibility and exclusion information
- outer-fold assignment
- paired M0 and M1 predictions

### Run manifests

Each stage records:

- input file hashes
- configuration hashes
- method parameters
- random seeds
- model and checkpoint provenance
- software environment
- exclusions and warnings

## Synthetic smoke test

The repository includes a deterministic synthetic dataset generator. It
does not represent a biological result, but it can verify that the complete
pipeline works.

Create the synthetic inputs:

```bash
python examples/make_synthetic_demo.py /tmp/residyn-demo
```

Run the pipeline:

```bash
mpnn-dfi all /tmp/residyn-demo/target.yaml
```

The smoke test exercises structure processing, DFI/DCI, DMS ingestion,
ProteinMPNN score import, conservation import, feature assembly, grouped
validation, and artifact generation.

## Tests

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

The current tests cover:

- Hessian symmetry
- six rigid-body modes
- pseudoinverse symmetry
- perturbation-direction normalization
- functional DCI orientation
- DFI rotation stability
- mmCIF parsing
- alternate-location selection
- sequence mapping
- DMS mutation parsing
- wild-type identity validation
- ProteinMPNN score calculation
- residue-grouped cross-validation

## Interpretation boundaries

ResiDYN can test whether residue flexibility adds predictive information
beyond the baseline model.

The Phase 1 MVP+ does not establish:

- that dynamics causally makes ProteinMPNN fail
- that DFI explains mutation-specific differences at the same residue
- a universal ProteinMPNN failure mechanism
- generalization across all protein families
- a population-level model failure rate
- that a five-protein pilot constitutes a broad benchmark

These claims require broader protein-level validation and additional
mechanistic evidence.

## Roadmap

Planned development includes:

- completion of the TEM-1 production run
- DHFR ligand-contact sensitivity
- functional DCI for TEM-1 and DHFR
- additional pilot-protein configurations
- ProteinMPNN training-exposure audits
- sequence-only model sensitivity analysis
- gap and structural-context sensitivity
- protein-level aggregation across assays
- broader preregistered validation if the pilot signal warrants expansion

## Related projects and resources

- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- [ProteinGym](https://github.com/OATML-Markslab/ProteinGym)
- [ConSurf](https://consurf.tau.ac.il/)
- [LigandMPNN](https://github.com/dauparas/LigandMPNN)
- [M-CSA](https://www.ebi.ac.uk/thornton-srv/m-csa/)

## Citation

ResiDYN is under active development and does not yet have an archival
citation.

If you use this repository before a formal release, cite the repository URL
and the specific commit used in your analysis:

```text
Inabinette, D. ResiDYN: Residue dynamics analysis of ProteinMPNN=E2=80=93DM=
S
residuals.
https://github.com/DavidInabinette/ResiDYN
```

A `CITATION.cff` file and versioned DOI should be added before the first
public research release.

## License

A project license has not yet been selected. Add an explicit license before
distributing or accepting external contributions.

