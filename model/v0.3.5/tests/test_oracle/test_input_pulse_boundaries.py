from events.inputs import make_input


def test_dense_pulse_values_match_declared_rising_and_falling_edges():
    x, driven = make_input("dense_burst", 16, 131)
    node = int(driven[0])

    assert x(0.0)[node] == 1.0
    assert x(0.02)[node] == 0.0
    assert x(0.125)[node] == 1.0
    assert x(0.145)[node] == 0.0
    assert x(0.25)[node] == 1.0
    assert x(0.27)[node] == 0.0


def test_sparse_pulse_is_off_at_width_and_on_at_next_period():
    x, driven = make_input("sparse_pulse", 16, 131)
    node = int(driven[0])

    assert x(0.019999)[node] == 1.0
    assert x(0.02)[node] == 0.0
    assert x(1.0)[node] == 1.0
