import pytest

torch = pytest.importorskip("torch")

from qcore import CHEMICAL_ACCURACY, MolecularProblem  # noqa: E402
from GQE import GQE  # noqa: E402


def test_gqe_h2():
    p = MolecularProblem.from_library("H2")
    r = GQE(p, iterations=60, batch_size=32, seed=0).run()
    assert abs(r.error) < CHEMICAL_ACCURACY
    assert abs(r.refined_error) < 1e-8
    assert r.energy >= p.e_fci - 1e-10


def test_gqe_reproducible():
    p = MolecularProblem.from_library("H2")
    a = GQE(p, iterations=10, batch_size=8, seed=3, refine=False).run()
    b = GQE(p, iterations=10, batch_size=8, seed=3, refine=False).run()
    assert a.sequence == b.sequence and a.energy == b.energy


@pytest.mark.slow
def test_gqe_lih_refined():
    p = MolecularProblem.from_library("LiH")
    r = GQE(p, seed=0).run()
    assert abs(r.refined_error) < CHEMICAL_ACCURACY
