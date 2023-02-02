import numpy as np
import pandas as pd

from virny.metrics.stability_metrics import compute_entropy


# ========================== Test compute_entropy ==========================
def test_compute_entropy_true1():
    uq_labels = pd.DataFrame([
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
        [0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    ])

    entropy_lst = np.apply_along_axis(compute_entropy, 1, uq_labels.transpose().values)
    assert entropy_lst.tolist() == [0.30463609734923813, 0.30463609734923813, 0.30463609734923813, 0.30463609734923813,
                                    0.30463609734923813, 0.30463609734923813, 0.30463609734923813, 0.30463609734923813,
                                    0.30463609734923813, 0.30463609734923813]


def test_compute_entropy_true2():
    uq_labels = pd.DataFrame([
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
    ])

    entropy_lst = np.apply_along_axis(compute_entropy, 1, uq_labels.transpose().values)
    assert entropy_lst.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_compute_entropy_true3():
    uq_labels = pd.DataFrame([
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
    ])

    entropy_lst = np.apply_along_axis(compute_entropy, 1, uq_labels.transpose().values)
    assert entropy_lst.tolist() == [0.5623351446188083, 0.5623351446188083, 0.5623351446188083, 0, 0, 0, 0, 0, 0 ,0]


def test_compute_entropy_true4():
    uq_labels = pd.DataFrame([
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 0, 1, 1, 0, 0],
    ])

    entropy_lst = np.apply_along_axis(compute_entropy, 1, uq_labels.transpose().values)
    assert entropy_lst.tolist() == [0.6931471805599453, 0.6931471805599453, 0.6931471805599453, 0.6931471805599453, 0.6931471805599453,
                                    0.6931471805599453, 0.6931471805599453, 0.6931471805599453, 0.6931471805599453, 0.6931471805599453,]
