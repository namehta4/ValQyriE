# ValQyriE

CPU-only, exact-statevector implementations of the **Variational Quantum Eigensolver (VQE)** and the
**Generative Quantum Eigensolver (GQE / GPT-QE)** for molecular ground-state energies.

## From molecule to ground-state energy

Both methods find the lowest electronic energy of a molecule. They share every step except the last:

```
 1  molecule            atoms, coordinates, charge, spin
 2  basis set           atomic orbital functions (STO-3G, 6-31G, ...)
 3  Hartree-Fock        molecular orbitals, reference state, E_HF
 4  active space        which orbitals go onto qubits
 5  qubit Hamiltonian   fermion-to-qubit mapping: H = Σ c_P P
 6  solve               VQE: ansatz U(θ) + classical optimizer
                        GQE: transformer generates circuits
 7  check               compare with exact E_FCI
```

1. **Molecule.** Atoms, their fixed positions (the Born-Oppenheimer approximation: nuclei don't move),
   total charge and spin. The library holds experimental equilibrium geometries; you can also give a
   bond length or your own coordinates.
2. **Basis set.** Electrons are described with a finite set of atomic orbital functions (STO-3G,
   6-31G, cc-pVDZ, ...). A larger basis is more accurate but gives more orbitals, and every orbital
   costs 2 qubits.
3. **Hartree-Fock (HF).** A mean-field calculation (PySCF RHF, or ROHF for open shells) gives
   molecular orbitals and the best single-determinant energy, E_HF. HF misses the *correlation
   energy*: the part of the energy that comes from electrons avoiding each other. Recovering that part
   is the job of the quantum algorithm. The HF determinant is also its starting state.
4. **Active space.** Only some orbitals are treated on qubits. Low-lying core orbitals stay filled and
   high-lying ones stay empty. Their effect is folded into the integrals (CASCI), so the problem
   shrinks to 2 × (active orbitals) qubits.
5. **Qubit Hamiltonian.** The electronic Hamiltonian, written with fermionic creation and
   annihilation operators, is mapped onto qubits (Jordan-Wigner, Bravyi-Kitaev or parity), one qubit
   per spin orbital. The result is a weighted sum of Pauli strings, H = Σ c_P P.
6. **Solve.** Find a quantum state |ψ⟩ with the lowest possible energy E = ⟨ψ|H|ψ⟩. The variational
   principle guarantees that E is never below the true ground-state energy, so lower is better.
   * **VQE** fixes a circuit shape, the *ansatz* U(θ) (e.g. UCCSD), applies it to the HF state and lets
     a classical optimizer (here BFGS with exact gradients) adjust the angles θ to minimise E.
     *ADAPT-VQE* builds the ansatz one operator at a time, always adding the one that lowers the energy
     fastest.
   * **GQE** has no circuit angles to optimise. A small transformer model (like a GPT language model)
     writes circuits as sequences of "words" from a fixed vocabulary of operators with preset angles.
     Each circuit's energy is evaluated, and the model is trained to make low-energy circuits more
     likely. The best circuit can then be fine-tuned with the VQE optimizer.
7. **Check.** In the active space the problem is small enough to solve exactly (full configuration
   interaction, FCI). Every result is reported as its error against E_FCI. The usual target is
   *chemical accuracy*, 1.6 mHa (1 kcal/mol).

On a quantum computer, step 6 would estimate E from repeated measurements, which adds statistical
noise and hardware errors. Here the quantum state is simulated exactly on the CPU, so energies and
gradients have no noise. The answer is exact for the chosen basis and active space. How close it is
to experiment depends on those two choices.

## Features

* Any popular ansatz: UCCSD, UCCD, UCCGSD, k-UpCCGSD, pUCCD, qubit-excitation UCCSD, hardware-efficient,
  symmetry-preserving gate fabric, LUCJ, SPA, ADAPT-VQE (fermionic / QEB / qubit / CEO pools,
  optionally TETRIS), or your own generators; any of them with orbital optimization (oo-VQE).
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
│   ├── excitations.py  fermionic / qubit excitation generators, operator pools, screening
│   ├── active_space.py opt-in MP2 active-space selection
│   ├── fci_interface.py qubit state <-> PySCF CI vector, density matrices
│   └── circuit.py      exact statevector circuits + adjoint-method gradients
├── VQE/              VQE, ansatze, ADAPT-VQE (CEO, TETRIS), orbital-optimized VQE, CLI (python -m VQE)
├── GQE/              GPT model, operator pool, GQE training loop, CLI (python -m GQE)
├── stress/           verified CPU stress test + system ceilings (python -m stress)
├── examples/         benchmark.py, dissociation_curve.py, stress/*.json case files
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
python -m VQE --molecule LiH --ansatz adapt --pool ceo --tetris -v   # TETRIS-CEO-ADAPT-VQE
python -m VQE --molecule N2 --ansatz kupccgsd --layers 2 --mapping bk
python -m VQE --molecule H2O --ansatz fabric                 # symmetry-preserving gate fabric
python -m VQE --molecule H6 --ansatz lucj --layers 2         # local unitary cluster Jastrow
python -m VQE --molecule H6 --bond-length 2.0 --ansatz spa --orbital-opt   # oo-SPA, vs CASSCF
python -m VQE --molecule H2 --ansatz hea --layers 3 --number-penalty 0.5 --spin-penalty 0.5
python -m VQE --molecule LiH --bond-length 2.5 --basis 6-31g --active 2,5
python -m VQE --molecule N2 --basis 6-31g --active 6,8 --screen  # drop symmetry-forbidden doubles
python -m VQE --molecule H2O --basis 6-31g --active auto        # opt-in: MP2 picks the active space
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
qubit (Tang 2021) and CEO (Ramôa et al., npj Quantum Inf. 2025, arXiv:2407.08696). Two newer
options come from the literature:

* **CEO pool** (`--pool ceo`, Jordan-Wigner only). The qubit excitations acting on the same four qubits
  (two of them for opposite-spin sets, three for same-spin sets) are combined into *coupled exchange
  operators*. The selected operator is either a one-parameter sum or difference θ(T_a ± T_b) (a 9-CNOT
  circuit), or, when more than one of the excitations on those qubits has a nonzero gradient, the
  excitations with one parameter each (13 CNOTs, the cost of a single qubit excitation). This follows
  the selection rule of the paper.
* **TETRIS** (`--tetris`, Anastasiou et al., arXiv:2209.10562). Each iteration adds, in order of
  gradient size, every operator with a nonzero gradient whose qubits don't overlap those already
  chosen. Operators on disjoint qubits run in parallel on hardware, so circuits get shallower and
  fewer iterations are needed (fewer rounds of gradient measurements on hardware). It suits the
  pools whose operators act on at most four qubits (`qeb`, `qubit`, `ceo`). Fermionic operators carry
  Jordan-Wigner strings and almost never have disjoint qubits.

`AdaptResult.cnot_count` adds up the CNOTs of the published circuit for every added operator: 2 for
a single qubit excitation, 13 for a double, 9 for a one-parameter CEO, 13 for a multi-parameter CEO,
and a CNOT ladder of 2(w − 1) for each Pauli string of weight w in the qubit and fermionic pools. It
is a hardware-cost estimate; the simulation itself applies every operator exactly.
`at_accuracy()` gives the iteration, parameters, CNOT count and CNOT depth of the shortest circuit
along the run that was within chemical accuracy. `cnot_depth` schedules each operator as early as its
qubits allow, which is where TETRIS gains.

**Which ADAPT pool?** Resources of the first circuit within chemical accuracy (CNOT count / CNOT
depth; ✗ = never reached), STO-3G, Jordan-Wigner, `grad_tol` 1e-3. The stretched geometries are the
ones used by Ramôa et al.:

| problem | qubits | fermionic | QEB | TETRIS-QEB | qubit | CEO | TETRIS-CEO |
|---|---|---|---|---|---|---|---|
| LiH, equilibrium | 12 | 400 / 400 | 65 / 55 | 91 / 33 | 30 / 30 | 45 / 35 | 63 / 21 |
| BeH2, equilibrium | 12 | 608 / 560 | 130 / 77 | 173 / 66 | 60 / 42 | 93 / 54 | 129 / 48 |
| H2O, equilibrium | 12 | 1168 / 1072 | 221 / 121 | 273 / 88 | 102 / 60 | 148 / 85 | 175 / 72 |
| LiH, 3.0 Å | 12 | 392 / 392 | 69 / 57 | 151 / 46 | 58 / 58 | 49 / 37 | 107 / 30 |
| BeH2, 2.0 Å | 12 | 1520 / 1472 | 279 / 178 | 320 / 110 | 258 / 210 | 164 / 96 | 213 / 75 |
| H6, 1.5 Å | 12 | 5536 / 5264 | 1318 / 879 | 1395 / 429 | ✗ | 787 / 617 | 862 / 295 |

* For LiH at 3.0 Å, TETRIS-CEO needs 107 CNOTs at depth 30, the same as Table I of Ramôa et al.
  (107 / 30). For H6 at 1.5 Å it needs 862 / 295, against the paper's 812 / 282, which also uses
  further measurement-saving techniques. (Our BeH2 freezes the Be 1s core, 12 qubits, so it is not
  comparable with the paper's 14-qubit result.)
* **CEO** is the best choice for hard (stretched, strongly correlated) problems: fewest CNOTs on
  BeH2 at 2.0 Å and H6 at 1.5 Å. **TETRIS** cuts the circuit depth by 2-3× and halves the iterations,
  for 10-40% more CNOTs. The **qubit** pool gives the fewest CNOTs on easy molecules near equilibrium,
  but needs many more iterations and fails on H6. **Fermionic** circuits are 5-10× more expensive
  in CNOTs; all pools reach similar final energies.
* Rerun with `python examples/benchmark.py --ansatz adapt-qeb adapt-ceo adapt-ceo-tetris`.

### Newer ansätze and orbital optimization

* **Gate fabric** (`fabric`; Anselmetti et al., New J. Phys. 23, 113010 (2021)). A hardware-style
  circuit that, unlike `hea`, keeps the electron number, S_z and S² exact. Each layer is a brick
  wall of 4-qubit gates on neighbouring orbitals: a pair exchange followed by a Givens orbital
  rotation. `--layers` defaults to the number of orbitals; fewer layers are much less accurate.
* **LUCJ** (`lucj`; Motta et al., Chem. Sci. 14, 11213 (2023)). The ansatz IBM uses for chemistry
  on its hardware: |ψ⟩ = Π e^K e^{iJ} e^{−K} |HF⟩. Here e^K is a general orbital rotation (a
  network of nearest-neighbour Givens rotations), and J holds density-density terms restricted to
  what the qubit layout connects (`--interactions square`, `heavy-hex` or `all`). It conserves N and
  S_z, but S² only approximately, as in the paper. The paper starts from CCSD amplitudes; here the
  parameters start small and random, so use 2 layers.
* **SPA, separable pair approximation** (`spa`; Kottmann & Aspuru-Guzik, Phys. Rev. A 105, 032449
  (2022)). Every electron pair gets its own orbitals, so the circuit is a product of independent
  pair functions: the fewest parameters and the shallowest circuit of all. Only accurate in good
  orbitals, so run it with `--orbital-opt`. The virtual orbitals are assigned to pairs automatically
  (largest exchange integral) or with `SPA(pairs=[[...], ...])`.
* **Orbital optimization** (`--orbital-opt`, `OrbitalOptimizedVQE`; Mizukami et al. 2020, Sokolov
  et al. 2020). The orbitals are optimized together with the circuit, i.e. CASSCF with VQE as the
  solver. It alternates VQE in the current orbitals with orbital steps driven by the VQE state's
  density matrices, using PySCF's analytic orbital gradient. Each orbital step is accepted only if
  it lowers the energy, so the energy never goes up. The reference is CASSCF, the exact answer for
  orbital-optimized methods. Pair ansätze (`puccd`, `spa`, `kupccgsd`, `lucj`) also rotate the
  active orbitals among themselves and start from Boys-localized orbitals (`--initial-orbitals`).

Error vs FCI in Ha (parameters), STO-3G, Jordan-Wigner, default settings. oo-pUCCD and oo-SPA
include orbital optimization; their error is against FCI in the Hartree-Fock orbitals, which for N2
lies 15 mHa above CASSCF:

| | H4, 1.0 Å | H4, 2.0 Å | H6, 2.0 Å | LiH | H2O | N2 |
|---|---|---|---|---|---|---|
| UCCSD | 8.0e-5 (26) | 4.6e-3 (26) | 1.3e-2 (117) | 1.1e-5 (92) | 1.0e-4 (92) | 4.1e-4 (117) |
| k-UpCCGSD | 1e-14 (24) | 1.8e-2 (24) | 2.4e-2 (60) | 4.4e-9 (60) | 7.8e-5 (60) | 1.7e-3 (60) |
| gate fabric | 3e-15 (24) | 3e-15 (24) | 4.5e-3 (60) | 2.2e-8 (60) | 1.6e-3 (60) | 1.8e-2 (60) |
| LUCJ, 1 layer | 4.5e-3 (35) | 1.0e-2 (35) | 2.6e-2 (77) | 2.2e-4 (77) | 3.0e-3 (77) | 1.8e-2 (77) |
| LUCJ, 2 layers | 4.3e-6 (70) | 5.1e-5 (70) | 5.1e-3 (154) | 1.5e-5 (154) | 7.7e-4 (154) | 7.2e-3 (154) |
| oo-pUCCD | 3.1e-2 (4) | 3.2e-2 (4) | 1.5e-2 (9) | 1.2e-4 (8) | 6.6e-3 (8) | 4.4e-2 (9) |
| oo-SPA | 3.7e-2 (2) | 3.2e-2 (2) | 1.5e-2 (3) | 1.2e-3 (4) | 9.4e-3 (2) | 4.5e-2 (3) |

**Which one should I use?**

| goal | ansatz |
|---|---|
| an accurate energy, standard method | `uccsd`, or ADAPT with `--pool ceo` |
| a circuit that could run on hardware and keeps all symmetries | `fabric` |
| IBM-style hardware studies (square / heavy-hex layouts) | `lucj --layers 2` |
| the fewest parameters, qualitative bond breaking | `spa` or `puccd` with `--orbital-opt` |
| better orbitals for any ansatz | add `--orbital-opt` |

The pair ansätze (pUCCD, SPA) treat electrons only in pairs, which limits them to about 1-5·10⁻² Ha
here. That accuracy is the price of their tiny circuits, not a bug. The gate fabric and LUCJ can get
stuck in local minima (N2 above), because they start from random parameters.

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

### Screening: should I turn it on?

Many double excitations in UCCSD can never lower the energy, because the molecule's symmetry forbids
them. Their best value is exactly zero, so they only add parameters and run time. **Screening** finds
and removes them before the run: a double is dropped when its energy gradient at the Hartree-Fock
state is zero (below 10⁻⁸). Singles are always kept.

**Simple rule:**

| you are running | screening | how |
|---|---|---|
| VQE with `uccsd`, `uccd` or `quccsd` to get an energy | **turn it on** | `python -m VQE ... --screen`, or `UCCSD(screen_tol=1e-8)` |
| GQE | leave it on (already the default) | `--no-screen` turns it off |
| ADAPT-VQE | not needed: ADAPT already picks only useful operators | — |
| `uccgsd`, `kupccgsd`, `puccd`, `hea` | not available (the test doesn't work for generalized excitations) | — |
| the stress test, comparing machines | **leave it off** (default): the full problem is more work per step, and results stay comparable | — |
| the stress test, mirroring a screened science run | turn it on | `"screen": true` in the `vqe` section |

The energy is the same with and without screening. Measured (UCCSD, BFGS) on nine library molecules
(LiH, BeH2, H2O, HF, N2, CO, NH3, CH4, O2), the energies agree to better than 2·10⁻⁹ Ha, while the
parameter count drops by 12–67%:

| problem | qubits | parameters (full → screened) | VQE time |
|---|---|---|---|
| H2O / STO-3G | 12 | 92 → 40 | 0.1 s → 0.1 s |
| N2 / STO-3G | 12 | 117 → 65 | 0.3 s → 0.2 s |
| NH3 / STO-3G | 14 | 204 → 152 | 3.5 s → 2.5 s |
| N2 / 6-31G, (6e, 8o) | 16 | 315 → 97 | 14.0 s → 8.6 s |

The savings grow with the basis set, so screening matters most for bigger problems. Small molecules
barely change, and an optimizer can even take a slightly different path and finish later (CH4: 2.5 s
→ 5.0 s).

**One caution.** Screening assumes the ground state has the same symmetry as the Hartree-Fock state.
That is true for the library molecules at their default geometries. It can fail for unusual cases,
such as strongly stretched bonds or some open-shell molecules. If you are unsure, run a small version
once with and once without `--screen`. Both runs should give the same energy.

### Basis sets

`--basis` (or `"basis"` in a stress case file) accepts any basis set built into PySCF, about 360 names
such as `sto-3g`, `sto-6g`, `3-21g`, `6-31g`, `6-31g*`, `6-311g`, `def2-svp`, `cc-pvdz` and
`aug-cc-pvdz`. An unknown name is rejected with an error. Installing `basis-set-exchange` (`pip
install basis-set-exchange`) makes PySCF look up names it doesn't have built in.

The basis decides how many orbitals the molecule has, and each orbital in the active space costs 2
qubits. A larger basis is more accurate but quickly exceeds what a CPU can simulate, so beyond STO-3G
you will almost always also need an active space:

| basis | H2 | LiH | H2O | N2 | CH4 |
|---|---|---|---|---|---|
| `sto-3g` (default) | 2 | 6 | 7 | 10 | 9 |
| `6-31g` | 4 | 11 | 13 | 18 | 17 |
| `6-31g*` | 4 | 16 | 18 | 28 | 22 |
| `cc-pvdz` | 10 | 19 | 24 | 28 | 34 |
| `aug-cc-pvdz` | 18 | 32 | 41 | 46 | 59 |

(numbers are orbitals; qubits = 2 × the orbitals kept in the active space)

### Active space

The active space is the set of orbitals that go onto qubits. All other orbitals are kept either
always filled (core) or always empty. Choices (`--active` on the command line, `"active_space"` in a
stress case file):

* `default` (**the default**): the hand-picked active space listed for each library molecule (≤ 14
  qubits in STO-3G).
* `full`: all electrons and orbitals.
* `NELEC,NORB` (`[NELEC, NORB]` in JSON): the NORB orbitals nearest the highest occupied one, by
  Hartree-Fock orbital energy.
* `auto` (**opt-in, never used unless asked for**): let MP2 pick the orbitals.

**How `auto` works.** A quick MP2 calculation estimates how filled each orbital really is.
Hartree-Fock says every orbital holds exactly 2 or 0 electrons; with electron correlation, some hold
for example 1.97 or 0.03. Orbitals that stay within `threshold` (default 0.01) of 2 or 0 barely take
part, so they are left out. The rest are kept, up to `max_qubits` (default 16). If too many qualify,
the most fractional ones win. Open-shell (singly occupied) orbitals are always kept. Orbitals of
equal energy, such as a pair of π orbitals, are kept or dropped together.

```bash
python -m VQE --molecule H2O --basis 6-31g --active auto --auto-max-qubits 12 --screen
```
```json
"vqe": {"molecule": "H2O", "basis": "6-31g", "active_space": {"method": "auto", "max_qubits": 12}}
```

The chosen orbitals and their MP2 occupations are printed. At the same qubit count it can do better
than picking orbitals by energy: H2O / 6-31G with 12 qubits gives E_FCI = −76.0056 Ha with `auto`
against −75.9951 Ha with the library's (8e, 6o). With few orbitals worth correlating it simply
picks a smaller space: LiH / STO-3G becomes 4 qubits instead of 12, which is cheaper but also less
accurate.

**When not to trust it.** MP2 is an approximation that breaks down exactly where electrons are
strongly correlated, such as stretched bonds (e.g. N2 at 2.0 Å). The tool then prints a WARNING. The
selected orbitals are usually still reasonable, but check against a hand-picked active space. A
WARNING is also printed when more orbitals qualify than `max_qubits` allows.

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
for qualifying cluster hardware (e.g. Student Cluster Competition nodes). The tool knows the correct
answer to every computation it runs and checks each result against it. A healthy machine passes every
check. Unstable hardware often doesn't crash: it silently returns a wrong number (*silent data
corruption*). Typical causes are a marginal CPU, a bad core, an aggressive overclock/undervolt or BIOS
power setting, overheating, or memory errors. Here such errors are reported as **failures**, with the
offending core named. An ordinary benchmark would only show them as a slightly different score.

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

### Case files (`--config`)

A run can be described in a JSON case file instead of on the command line. This is the only place
to set the VQE molecule's bond length, basis set and active space. The file's top-level keys are the
`run` options (`duration`, `mode`, `workloads`, `size`, `workers`, `qubits`, `dgemm_n`, `interval`,
`no_peak`, `peak_seconds`, `output`, `freq_ghz`, `flops_per_cycle`, `mem_bw_gbs`, `mpi`; dashes or
underscores). An optional `vqe` section describes the molecule. Options given on the command line
override the file, so one case file can serve many runs:

```bash
python -m stress run --config examples/stress/n2_stretched_631g.json
python -m stress run --config examples/stress/burnin.json --duration 600   # shorter than the file says
```

```json
{
  "mode": "mixed", "duration": 3600, "size": "dram", "output": "burnin-report.json",
  "vqe": {"molecule": "H2O", "bond_length": 1.1, "basis": "6-31g", "active_space": [8, 8]}
}
```

The `vqe` section takes one of two forms:

* **Library molecule**: `molecule` (H2, HeH+, LiH, BeH2, H2O, NH3, CH4, HF, N2, CO, O2, H4, H6,
  H4-ring; case-insensitive), with optional `bond_length` (Å), `basis` (any PySCF basis name),
  `active_space` (`"default"`, `"full"`, `[n_electrons, n_orbitals]`, or the opt-in `"auto"` /
  `{"method": "auto", "threshold": 0.01, "max_qubits": 16}`; see [Active space](#active-space)) and
  `mapping` (`jw`, `bk` or `parity`).
* **Any other molecule**: `geometry` (a PySCF atom string in Å), with optional `charge`, `spin`
  (N_α − N_β), `basis`, `active_space` (`"full"`, `[n_e, n_o]` or `"auto"`), `mapping` and `name`. See
  [`examples/stress/custom_geometry.json`](examples/stress/custom_geometry.json) (ethylene).

Either form also takes `"screen": true`, which removes symmetry-forbidden doubles from the UCCSD
circuit. Leave it off (the default) when comparing machines; see
[Screening](#screening-should-i-turn-it-on).

The JSON report records the full case file, so a run can be reproduced from it. Unknown keys, invalid
values, an unknown molecule or basis, and an impossible active space are all rejected with a message
before any load starts. A problem too big for RAM is rejected the same way: each worker holds its own
copy of the sparse Hamiltonian plus about 128 B per amplitude. The tool prints the per-worker memory
and refuses a problem that needs more than 80% of RAM across all workers.

`--molecule NAME` on the command line is a shortcut for a library molecule at default settings, and
replaces the file's `vqe` section. Bigger problems mean more work per step. Measured on one Apple M4
core:

| problem | qubits | UCCSD parameters | time per step | memory per worker |
|---|---|---|---|---|
| H2O / STO-3G, default (8e, 6o) | 12 | 92 | 5 ms | ~2 MiB |
| NH3 or CH4 / STO-3G, default | 14 | 204 | 40 ms | 40–50 MiB |
| N2 / 6-31G, (6e, 8o) | 16 | 315 | 0.6 s | 87 MiB |
| H2O / 6-31G, (8e, 8o) | 16 | 360 | 2 s | 95 MiB |
| LiH / 6-31G, (2e, 9o) | 18 | 80 | 4 s | ~1 GiB |

H2 and HeH+ (4 qubits) finish a step in microseconds, so they measure Python overhead more than the
CPU and are not useful as a stress load.

### Defaults

| setting | default |
|---|---|
| `duration` | 60 s per phase (`phases` runs the four workloads one after another, so about 4 min) |
| `mode` | `phases` |
| `workloads` | `dgemm,statevector,vqe,transformer` (`transformer` is skipped if PyTorch is missing) |
| `workers` | `physical`: one worker per physical core, pinned on Linux |
| `size` | `auto` |
| ceiling measurement | on, `peak_seconds` 4 per measurement (`no_peak` to skip) |
| `interval` | 10 s between status lines and telemetry samples |
| theoretical peak | max boost clock from the OS and FLOPs/cycle from the ISA; memory bandwidth not set |
| `output` | none (no JSON report) |

Working-set sizes picked by `size`:

| workload | `cache` | `auto` | `llc` | `dram` |
|---|---|---|---|---|
| `dgemm` matrix n | 256 | 1024 | 1024 | 2048 |
| `statevector` qubits¹ | fits in L2 | ≤ 24 | fits in L3 | ≤ 26 |
| `vqe` molecule | H4 | H2O | H2O | NH3 |
| `transformer` width | 128 | 256 | 256 | 512 |

¹ Never below 10, and never above what fits in half the RAM across all workers. Bigger states use
fewer rotations per step (16 at 20 qubits and below, halving for each extra qubit, minimum 2), which
keeps each step to about a second.

Defaults for the `vqe` section are the library's experimental bond length, `sto-3g`, the library's
active space (all orbitals for a custom `geometry`), `jw`, charge 0, spin 0 and no screening. The
circuit is always UCCSD at fixed random parameters (0.05 · N(0, 1), seed 7). The transformer is the GQE GPT with 4
layers, 8 heads, sequence length 64, batch 8 and vocabulary 512.

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

**Correctness vs. maximum load.** This tool checks that the hardware *computes correctly* under the
real VQE/GQE workload at your chosen clocks and BIOS settings. It is not designed to reach the
highest possible power draw. Python/NumPy overhead keeps its power and heat below those of HPL or
Prime95. Use both: HPL (which SCC runs anyway) shows the cluster survives the thermal and power worst
case, and `python -m stress run` shows it still gives correct answers. On Apple silicon, BLAS runs on the AMX/SME matrix unit, so measured GEMM can
exceed the vector-ISA estimate; the tool says so when it happens.

## Limits and scope

* Exact statevector: memory 16·2ⁿ bytes and Hamiltonians as sparse matrices. ≤ 16 qubits is
  comfortable on a laptop, ~20 is feasible. The default active spaces keep the library at ≤ 14 qubits.
* Results are noiseless and use infinite shots: no hardware noise, measurement sampling or
  readout error.
* GQE is stochastic. Results depend on the seed and the training budget, so report several seeds.
