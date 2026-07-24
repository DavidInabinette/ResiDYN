# Legacy DFI/DCI source

The supplied `all-dci`-style script is treated as an external provenance input,
not as production code. It contains two behaviorally important defects:

1. `calc_covariance(..., Verbose)` passes `Verbose` positionally into
   `calchessian(..., cutoff=None, Verbose=False)`, so a true verbosity flag can
   silently become a 1 Å cutoff.
2. `calcperturbMat` divides by the literal value seven instead of the number of
   supplied directions.

The implementation in `mpnn_dfi.dfi` reproduces the intended `d^-6` spring law,
uses keyword-only options, divides by the actual direction count, records its
tolerances, and exposes the response-matrix orientation explicitly.

Keep the original script unchanged beside the raw analysis inputs and record
its checksum in the study manifest if a legacy-comparison run is performed.

