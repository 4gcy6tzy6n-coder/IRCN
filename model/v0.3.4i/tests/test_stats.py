from ircn.stats import holm_adjust


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    adjusted = holm_adjust({"a": 0.001, "b": 0.02, "c": 0.03})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert adjusted["a"] >= 0.003
    assert adjusted["c"] == 0.04
