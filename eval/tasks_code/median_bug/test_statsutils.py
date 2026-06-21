from statsutils import mean, median


def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5
    assert median([10, 20]) == 15


def test_mean():
    assert mean([2, 4, 6]) == 4
