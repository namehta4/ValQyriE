"""The plug-in interface every implementation provides, and the reference (baseline) one.

An implementation is a Python class (it may wrap C++, Fortran, Kokkos, SYCL, CUDA/HIP ... through
pybind11, ctypes, cffi or similar):

    class MyEvaluator:
        def __init__(self, case: bench.cases.CaseData, comm=None): ...
        def energy_and_gradient(self, theta: numpy.ndarray) -> tuple[float, numpy.ndarray]: ...

``case`` holds the integrals and the excitation list (see ``bench.cases`` for the exact
definition). ``comm`` is an mpi4py communicator under ``--mpi`` (else None); every rank calls
``energy_and_gradient`` with the same theta and must return the same, complete result.
Setup work (building tables, uploading to a GPU) belongs in ``__init__``: it is timed
separately from the evaluations.
"""
from __future__ import annotations

import importlib

import numpy as np


class ReferenceEvaluator:
    """Stage 1.1 baseline: the ValQyriE code as it stands (full 2^n statevector, NumPy).

    H psi uses the sparse Hamiltonian up to 20 qubits and the matrix-free Pauli-sum product
    above (a sparse matrix in the full space would need hundreds of GB at 24 qubits).
    """

    def __init__(self, case, comm=None):
        from qcore.circuit import Circuit
        from qcore.excitations import excitation_generator
        from qcore.problem import MolecularProblem
        p = MolecularProblem.from_integrals(case.h1, case.h2, case.e_core, case.n_alpha, case.n_beta, "jw",
                                            e_fci=np.nan)
        circ = Circuit(p.n_qubits, case.hf_index)
        for k, exc in enumerate(case.excitations):
            circ.add_generator(excitation_generator(p.mapping, exc), circ.new_parameter(f"t{k}"))
        self.circuit = circ
        if p.n_qubits <= 20:
            self.hamiltonian = p.hamiltonian_sparse
        else:
            from scipy.sparse.linalg import LinearOperator
            ham = p.hamiltonian
            dim = 1 << p.n_qubits
            self.hamiltonian = LinearOperator((dim, dim), matvec=ham.apply, dtype=np.complex128)

    def energy_and_gradient(self, theta):
        return self.circuit.energy_and_gradient(np.asarray(theta, float), self.hamiltonian)


def load_impl(spec: str | None):
    """``module:Class`` (importable, or a path ``dir/file.py:Class``); None = the reference."""
    if not spec:
        return ReferenceEvaluator
    mod_name, _, cls_name = spec.partition(":")
    if not cls_name:
        raise ValueError(f"--impl must look like module:Class, got {spec!r}")
    if mod_name.endswith(".py"):
        import importlib.util
        import sys
        from pathlib import Path
        path = Path(mod_name).resolve()
        sys.path.insert(0, str(path.parent))
        spec_ = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_name)
    return getattr(module, cls_name)
