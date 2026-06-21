def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2]  # BUG: wrong for even-length lists (should average the two middle values)


def mean(xs):
    return sum(xs) / len(xs)
