"""Demo fixture for `atelier agent`. The bug below is intentional — the agent is
meant to find and fix it. Reset anytime with: git checkout sample_task/"""


def add(a, b):
    return a - b  # BUG: should be a + b


def is_even(n):
    return n % 2 == 0
