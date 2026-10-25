# ValQyriE

CPU-only, exact-statevector implementations of the **Variational Quantum Eigensolver (VQE)** and the
**Generative Quantum Eigensolver (GQE / GPT-QE)** for molecular ground-state energies.

* Any popular ansatz: UCCSD, UCCD, UCCGSD, k-UpCCGSD, pUCCD, qubit-excitation UCCSD, hardware-efficient,
  ADAPT-VQE (fermionic / QEB / qubit pools), or your own generators.
* Any of Jordan-Wigner, Bravyi-Kitaev or parity encodings.
* Benchmark molecules (H2, HeH+, LiH, BeH2, H2O, NH3, CH4, HF, N2, CO, O2 triplet, H4/H6 chains, H4 ring),
  any PySCF basis, any active space, or any geometry you provide.
* Every result is validated against exact diagonalisation (FCI within the active space).
* Pure Python (NumPy / SciPy / PySCF / PyTorch-CPU) with no compiled code of its own and nothing tied
  to one CPU model. The NumPy/SciPy/PyTorch wheels pick SSE/AVX2/AVX-512/NEON kernels at runtime, so
  one install runs on AMD EPYC, Intel Xeon, ARM servers and Apple silicon (Linux, macOS; Windows via
  WSL2). The test suite passes on arm64 macOS and on x86_64 Linux, the latter run under emulation.
  NumPy 1.24+ and 2.x are both supported.

```
ValQyriE/
├── qcore/            shared core (used by both solvers)
│   ├── pauli.py        vectorised Pauli algebra (symplectic X^x Z^z form), sparse matrices
│   ├── mappings.py     JW / BK / parity as GF(2) linear encodings
│   ├── hamiltonian.py  second-quantised H -> qubit H; N, S_z, S^2 operators
│   ├── problem.py      MolecularProblem: PySCF RHF/ROHF, CASCI active space, references
│   ├── molecules.py    molecule library (experimental r_e geometries)
│   ├── excitations.py  fermionic / qubit excitation generators, operator pools
│   └── circuit.py      exact statevector circuits + adjoint-method gradients
├── VQE/              VQE, ansatze, ADAPT-VQE, CLI (python -m VQE)
├── GQE/              GPT model, operator pool, GQE training loop, CLI (python -m GQE)
├── stress/           verified CPU stress test + system ceilings (python -m stress)
├── examples/         benchmark.py, dissociation_curve.py
└── tests/            validation suite (pytest)
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
# Linux: install CPU-only PyTorch first, which avoids the multi-GB CUDA build (macOS wheels are already CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[gqe,test]"        # add ,crosscheck for the OpenFermion cross-check
pytest -q -m "not slow"
```

Alternatives: `conda env create -f environment.yml` (conda-forge), or the multi-arch Docker image:
`docker buildx build --platform linux/amd64,linux/arm64 -t valqyrie .`.
VQE alone does not need PyTorch: `pip install -e .`.

## Quick start

```bash
python -m VQE --molecule LiH --ansatz uccsd                 # UCCSD, Jordan-Wigner
python -m VQE --molecule H2O --ansatz adapt --pool qeb -v   # ADAPT-VQE with qubit-excitation pool
python -m VQE --molecule N2 --ansatz kupccgsd --layers 2 --mapping bk
python -m VQE --molecule H2 --ansatz hea --layers 3 --number-penalty 0.5 --spin-penalty 0.5
python -m VQE --molecule LiH --bond-length 2.5 --basis 6-31g --active 2,5
python -m GQE --molecule LiH -v                             # GPT-QE, then VQE-refine the best circuit
python examples/benchmark.py --ansatz uccsd kupccgsd adapt --gqe
```

```python
from qcore import MolecularProblem
from VQE import VQE, AdaptVQE, UCCSD, KUpCCGSD, HardwareEfficient, CustomAnsatz
from GQE import GQE

p = MolecularProblem.from_library("H2O", mapping="parity")          # or MolecularProblem(geometry, basis, ...)
print(VQE(p, UCCSD(reps=2), optimizer="L-BFGS-B").run())
print(AdaptVQE(p, pool="fermionic", grad_tol=1e-3).run())
print(VQE(p, CustomAnsatz(["X0 Y1", "Y0 X1 X2 X3"])).run())       # exp(i θ σ) per string
print(GQE(p, iterations=300, seed=1).run())
```

## How it works

### Hamiltonian
PySCF runs RHF (closed shell) or ROHF (open shell). `mcscf.CASCI` then supplies the active-space
effective one-electron integrals, the frozen-core energy, and the exact CASCI (FCI-in-active-space)
reference energy. With chemist-notation integrals and interleaved spin orbitals (2k = α, 2k+1 = β), the
code builds

H = E_core + Σ h_PQ a†_P a_Q + ½ Σ (PR|QS) a†_P a†_Q a_S a_R

by mapping each `E_PQ = a†_P a_Q` to Pauli operators. Products of these are formed vectorised in the
symplectic representation, where `(X^x1 Z^z1)(X^x2 Z^z2) = (-1)^{|z1∧x2|} X^{x1⊕x2} Z^{z1⊕z2}` is exact.
Each encoding is defined by its binary matrix β (q = βn mod 2). The Majorana operators come out
generically as `c_p = Z^{parity set} X^{update set}` and `d_p = i c_p Z^{occupation set}`
(Seeley, Richard & Love 2012). The Fenwick-tree BK therefore works for any qubit count.

Validation (in `tests/`): canonical anticommutation relations, ⟨HF|H|HF⟩ = E_SCF and λ_min = E_CASCI to
1e-12 Ha for every mapping, the 15-term H2 and 631-term LiH Hamiltonians, and ⟨S²⟩ = 2 for triplet O2.
`tests/test_openfermion.py` runs when `pip install -e ".[crosscheck]"` is installed. It hands the same
integrals to OpenFermion and checks the following against our code:
* term-by-term agreement (≤ 1e-10) of the JW, BK (`bravyi_kitaev`) and parity (binary-code) Hamiltonians;
* every UCCSD excitation generator;
* identical spectra against OpenFermion's Fenwick-tree BK, which uses a different tree layout;
* ground-state energies from OpenFermion's own sector solver.

### Simulator
Every fermionic or qubit excitation generator G = T − T† is anti-Hermitian, and its Pauli strings
commute and share a single X mask. Hence G = D·X^x with D diagonal, G² = −|D|², and
`exp(θG) = cos(θ|D|) + sin(θ|D|)/|D| · G` exactly: a 2×2 rotation on each basis pair (b, b⊕x). Each
excitation is applied as one vectorised operation with no Trotter error inside an excitation.
(Different excitations are applied in a fixed product order, the standard "disentangled" UCC form.)
Gradients use the adjoint method: exact, and about the cost of three circuit evaluations for any
number of parameters. Tests check them against finite differences and `scipy.linalg.expm`.

### VQE
`VQE(problem, ansatz, optimizer=...)` accepts any `scipy.optimize.minimize` method (BFGS by default,
using exact gradients) or the built-in Adam / SPSA. Optional penalties μ(N−N₀)² and μ(S_z−S_z₀)² keep
symmetry-breaking ansätze (HEA) in the right particle-number/spin sector; reported energies exclude
the penalties. Results report the error vs FCI, the fraction of correlation energy recovered, ⟨S²⟩,
⟨N⟩, and a CNOT-count estimate.

ADAPT-VQE follows Grimsley et al. 2019: it selects the pool operator with the largest |⟨[H, A]⟩|,
then re-optimises all parameters with warm starts. Pools: fermionic, generalized, QEB (Yordanov 2020),
qubit (Tang 2021).

### GQE (GPT-QE)
This follows Nakaji et al., arXiv:2401.09253. The vocabulary is {(G_k, t)}: excitation generators ×
a symmetric grid of evolution times, so token j means U_j = exp(t_j G_j). A causal transformer
(PyTorch, CPU) samples length-L sequences with p ∝ exp(−β w), where w are the model's logits. Each
circuit's energy is evaluated exactly. The model is trained by logit matching
`(exp(−w_sum) − exp(−s(E − E_HF)))²`, while β is annealed linearly and an elite replay buffer is used.
No circuit parameters are optimised during generation. `refine=True` (the default) then optimises the
continuous angles of the best circuit with BFGS, and the result reports both numbers separately.
Doubles whose HF gradient is zero by symmetry are screened out of the pool (`screen_tol=None`
disables this).

## Benchmarks (STO-3G, Jordan-Wigner, Apple M-series CPU)

Error vs FCI in Ha (number of parameters). ✗ = outside chemical accuracy (1.6 mHa).
Reproduce with `python examples/benchmark.py --ansatz uccsd kupccgsd adapt`.

| Molecule | Qubits | E_FCI (Ha) | UCCSD | k-UpCCGSD (k=2) | ADAPT-VQE (fermionic) |
|---|---|---|---|---|---|
| H2 | 4 | -1.1372701747 | 2.2e-15 (3) | 2.9e-15 (4) | 2.2e-15 (1) |
| HeH+ | 4 | -2.8514676862 | -2.7e-15 (3) | -4.4e-16 (4) | -1.8e-15 (3) |
| LiH | 12 | -7.8824034103 | 1.1e-05 (92) | 1.1e-07 (60) | 1.1e-05 (33) |
| BeH2 | 12 | -15.5948379588 | 3.7e-04 (92) | 6.5e-08 (60) | 3.7e-04 (18) |
| H2O | 12 | -75.0125066442 | 1.0e-04 (92) | 7.8e-05 (60) | 9.7e-05 (30) |
| NH3 | 14 | -55.5190321544 | 1.7e-04 (204) | 2.5e-03 (84) ✗ | 1.7e-04 (140) |
| CH4 | 14 | -39.7706604789 | 2.6e-05 (204) | 5.9e-04 (84) | 3.1e-05 (150) |
| HF | 10 | -98.5965658355 | 0.0e+00 (24) | 5.3e-13 (40) | -7.1e-14 (10) |
| N2 | 12 | -107.6218488599 | 4.1e-04 (117) | 1.7e-03 (60) ✗ | 3.9e-04 (47) |
| CO | 12 | -111.3309465596 | 5.9e-03 (117) ✗ | 3.4e-03 (60) ✗ | 2.5e-04 (111) |
| O2 | 12 | -147.7239300106 | 3.5e-04 (68) | 1.6e-04 (60) | 3.5e-04 (16) |
| H4 | 8 | -2.1663874486 | 8.0e-05 (26) | 9.8e-15 (24) | 8.5e-05 (14) |
| H4-ring | 8 | -1.9151065495 | 1.1e-02 (26) ✗ | 1.9e-03 (24) ✗ | 1.7e-01 (6) ✗ |

Notes on the failures. The CO and H4-ring results are reproducible across random
initialisations and optimisers:
* **CO**: correlation is strong (0.106 Ha in the active space). Single-reference UCCSD recovers 94%;
  ADAPT-VQE grows past it to chemical accuracy.
* **Square H4**: the RHF reference is degenerate. UCCSD fails here (a known textbook case), and ADAPT
  stops in a local minimum where every pool gradient falls below the tolerance.
* **NH3 / N2 k-UpCCGSD**: k = 2 falls short here (limited expressivity and/or local minima). Try more
  layers (`--layers`); this was not tested further.

GQE (defaults: 300 iterations × 64 circuits, three seeds each). Error vs FCI in Ha, as best sampled
circuit → after angle refinement:

| Molecule | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| H2  | 5.2e-07 → 2e-15 | | |
| LiH | 8.0e-04 → 2.3e-04 | 1.0e-03 → 7.4e-04 | 2.6e-03 → 2.1e-04 |
| BeH2 | 1.9e-03 → 6.5e-04 | 1.7e-03 → 2.5e-04 | 1.4e-03 → 2.5e-04 |
| H2O | 7.3e-03 → 1.9e-03 | 5.8e-03 → 7.9e-04 | 9.5e-03 → 1.3e-03 |

Each GQE run takes about 70 s on one CPU thread.

## CPU stress test and system ceilings (`stress/`)

`python -m stress` turns the solvers' numerical kernels into a **self-verifying CPU stress test**, built
for qualifying cluster hardware (e.g. Student Cluster Competition nodes). Every result is checked
exactly or against physics, so a marginal CPU, a bad core, an unstable overclock/undervolt, overheating
or memory errors show up as **failures** rather than as a slightly lower benchmark number.

```bash
python -m stress info                         # hardware, ISA, runtime SIMD, estimated ceilings (no load)
python -m stress peak                         # measure what the system can achieve at most
python -m stress run --duration 300           # 4 workloads x 5 min, one pinned worker per physical core
python -m stress run --mode mixed --duration 3600 --size dram -o burnin.json   # 1 h burn-in
mpirun -np 4 --map-by ppr:1:node python -m stress run --mpi -o cluster.json    # 4 nodes, one rank each
```

The exit code is 0 for PASS/WARN and 1 for FAIL, so runs can be scripted (e.g. after every BIOS change).

### Workloads

All workers get identical inputs, so their results must also agree **across cores and nodes**. A core
whose bit-exact fingerprint differs from the majority is named in the report.

| workload | stresses | verification on every step |
|---|---|---|
| `dgemm` | FP64 FMA units via BLAS (AVX2/AVX-512/NEON/SVE) | small-integer matrices make A·B exact in FP64, checked exactly with integer Freivalds products (independent of BLAS) |
| `statevector` | irregular full-state gathers: caches, DRAM, TLB | random circuit U then U†: the state must come back to within 1e-10 in norm |
| `vqe` | the production UCCSD energy + adjoint-gradient path (sparse matvec, complex arithmetic) | energy/gradient vs reference ≤ 1e-9, variational bound E ≥ E_FCI |
| `transformer` | FP32 GEMM / oneDNN or Accelerate (the GQE model) | bit-identical loss and gradients on every step, finite values |

Each workload is also checked against its own first result, bit for bit. `--size cache|llc|dram`
chooses working sets that fit in L2, fit in L3, or spill to DRAM. `--mode phases` runs one workload
at a time to give clean per-workload numbers; `--mode mixed` has every core cycle through all of them
for maximum variation in thermal and power load.

### "What the system can achieve at most"

`peak` (also run at the start of `run` unless `--no-peak`) reports:

* **Theoretical FP64/FP32 peak** = physical cores × max clock × FLOPs/cycle. FLOPs/cycle is derived
  from the ISA and microarchitecture, for example:
  * AVX-512 with 2 FMA units: 32;
  * AMD Zen 4: 16;
  * AVX2: 16;
  * Neoverse V2: 16.

  This is an upper bound. Clocks under heavy AVX load sit below max boost, so pass the all-core AVX
  clock as `--freq-ghz`, and use `--flops-per-cycle` for SKUs the table doesn't know.
* **Measured ceilings**:
  * multithreaded BLAS DGEMM, one DGEMM per core (single core and all cores), and PyTorch FP32 GEMM;
  * triad memory bandwidth on one core and on all cores, counting the 40 B per element NumPy actually
    moves. To get percent-of-theoretical, pass `--mem-bw-gbs` = sockets × channels × MT/s × 8 B
    (e.g. EPYC 9004, 12 × DDR5-4800 ≈ 461 GB/s per socket).
* **Largest problem the node can hold** (80% of RAM): statevector qubits for the state alone, for this
  simulator, and for VQE with its sparse Hamiltonian.
* Each stress workload's throughput as a percentage of those ceilings.

On Linux the run also samples clock speed, temperature (`coretemp`/`k10temp` hwmon) and package power
from RAPL, which needs root on most kernels. Everything degrades gracefully when the data isn't readable.

### Using it for SCC

1. Run `info` and `peak` on every node first. Nodes of the same SKU should agree to within a few
   percent; an outlier usually means a DIMM in the wrong slot (fewer memory channels), BIOS
   power/C-state settings, or thermal throttling.
2. Do a `--mode mixed` burn-in of ≥ 1 h after every hardware, BIOS, firmware or tuning change. Any FAIL
   or "results differ" warning disqualifies that configuration.
3. Under a power cap, compare GFLOP/s per watt (needs root for RAPL) across BIOS settings, frequency
   caps and SMT (`--workers logical`).
4. Workers allocate their arrays after pinning, so on multi-socket and NUMA nodes memory lands on
   the local node through first-touch allocation.

This tool is not a maximum-power burner like HPL or Prime95: NumPy's multi-pass kernels draw less
power. Use it to find wrong answers and stability problems; use HPL (which SCC runs anyway) for the
thermal worst case. On Apple silicon, BLAS runs on the AMX/SME matrix unit, so measured GEMM can
exceed the vector-ISA estimate; the tool says so when it happens.

## Limits and scope

* Exact statevector: memory 16·2ⁿ bytes and Hamiltonians as sparse matrices. ≤ 16 qubits is
  comfortable on a laptop, ~20 is feasible. The default active spaces keep the library at ≤ 14 qubits.
* Results are noiseless and use infinite shots: no hardware noise, measurement sampling or
  readout error.
* GQE is stochastic. Results depend on the seed and the training budget, so report several seeds.
