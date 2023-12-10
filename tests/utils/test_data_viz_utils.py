import pandas as pd

from virny.utils.data_viz_utils import rank_with_tolerance


def test_rank_with_tolerance_true1():
    tolerance = 0.005
    pd_series = pd.Series([0.025, 0.027, 0.001, 0.002]) # should be only positive numbers
    expected_ranks = [2, 2, 1, 1]
    actual_ranks = rank_with_tolerance(pd_series, tolerance)
    assert actual_ranks.tolist() == expected_ranks


def test_rank_with_tolerance_true2():
    tolerance = 0.005
    pd_series = pd.Series([0.429, 0.289, 0.377, 0.259]) # should be only positive numbers
    expected_ranks = [4, 2, 3, 1]
    actual_ranks = rank_with_tolerance(pd_series, tolerance)
    assert actual_ranks.tolist() == expected_ranks


def test_rank_with_tolerance_true3():
    tolerance = 0.005
    pd_series = pd.Series([0.313, 0.157, 0.274, 0.147]) # should be only positive numbers
    expected_ranks = [4, 2, 3, 1]
    actual_ranks = rank_with_tolerance(pd_series, tolerance)
    assert actual_ranks.tolist() == expected_ranks


def test_rank_with_tolerance_true4():
    tolerance = 0.005
    pd_series = pd.Series([0.001, 0.001, 0.0, 0.0]) # should be only positive numbers
    expected_ranks = [1, 1, 1, 1]
    actual_ranks = rank_with_tolerance(pd_series, tolerance)
    assert actual_ranks.tolist() == expected_ranks


def test_rank_with_tolerance_true5():
    tolerance = 0.01
    pd_series = pd.Series([0.099, 0.092, 0.075, 0.085]) # should be only positive numbers
    expected_ranks = [2, 2, 1, 1]
    actual_ranks = rank_with_tolerance(pd_series, tolerance)
    assert actual_ranks.tolist() == expected_ranks
