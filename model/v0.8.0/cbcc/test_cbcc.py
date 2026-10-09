import numpy as np
from dataclasses import replace

from cbcc_tasks import generate, N_TICKS
from cbcc_model import RecurrentModel, WIDTH, rollout_full
from cbcc_counterfactuals import online_features, paired_labels, joint_benefit
from cbcc_scheduler import POLICIES, choose, fit_influence, mac_costs
from cbcc_evaluation import power_n, holm


def test_task_generation_is_deterministic_and_split_safe():
    a = generate("sparse_key", 31, 4)
    b = generate("sparse_key", 31, 4)
    c = generate("sparse_key", 32, 4)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
    assert not np.array_equal(a.x, c.x)
    assert a.x.shape == (N_TICKS, 32)


def test_paired_counterfactuals_are_finite_and_branch_local():
    model = RecurrentModel()
    seq = generate("sparse_key", 44, 0)
    features, labels = paired_labels(model, seq)
    assert features.shape == (24 * 4, 9)
    assert labels.shape == (96,) and np.isfinite(labels).all()
    states, _ = rollout_full(model, seq)
    f1 = online_features(model, seq, states, 3, 0)
    altered_future = replace(seq, x=seq.x.copy(), y=seq.y.copy())
    altered_future.y[10:] += 1000
    f2 = online_features(model, altered_future, states, 3, 0)
    assert np.array_equal(f1, f2)


def test_scheduler_determinism_budget_and_no_candidate_precomputation():
    model = RecurrentModel()
    seq = generate("feedback_switch", 45, 0)
    states = np.zeros((N_TICKS, 32, WIDTH))
    estimator = fit_influence(np.ones((10, 9)), np.arange(10, dtype=float))
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)
    for policy in POLICIES:
        a = choose(policy, model, estimator, seq, states, 4, 2 * mac_costs(model, estimator)[0], rng1)
        b = choose(policy, model, estimator, seq, states, 4, 2 * mac_costs(model, estimator)[0], rng2)
        assert a == b
        assert a[1] <= 2 * mac_costs(model, estimator)[0]


def test_equal_budget_uses_one_circuit_update_and_estimator_cost_is_counted():
    model = RecurrentModel()
    seq = generate("sparse_key", 46, 0)
    seq.x[3, :8] = 1.0
    states = np.zeros((N_TICKS, 32, WIDTH))
    estimator = np.r_[np.zeros(9), np.ones(9), np.asarray([0, 0, 0, 0, 10, 0, 0, 0, 0])]
    budget = mac_costs(model, estimator)[0] + mac_costs(model, estimator)[1]
    for policy in POLICIES:
        chosen, spent = choose(policy, model, estimator, seq, states, 3, budget,
                               np.random.default_rng(1))
        assert len(chosen) == 1
        assert spent <= budget
        score_cost = {"activity": 32, "no_influence": 36}.get(policy, 0)
        expected = budget if policy == "cbcc" else mac_costs(model, estimator)[0] + score_cost
        assert spent == expected


def test_activity_scoring_does_not_compute_skipped_candidate_states():
    model = RecurrentModel()
    seq = generate("sparse_key", 47, 0)
    seq.x[0, :8] = 0.5
    states = np.zeros((N_TICKS, 32, WIDTH))
    estimator = np.zeros(3 * 9)
    def forbidden(*args, **kwargs):
        raise AssertionError("candidate recurrent state was computed during scoring")
    model.step_circuit = forbidden
    chosen, _ = choose("activity", model, estimator, seq, states, 0, 3000,
                       np.random.default_rng(0))
    assert len(chosen) == 1


def test_activity_policy_never_updates_quiet_circuits_with_accumulated_credit():
    model = RecurrentModel()
    seq = generate("sparse_key", 471, 0)
    seq.x[0, :8] = 0.5
    states = np.zeros((N_TICKS, 32, WIDTH))
    estimator = np.zeros(27)
    update_mac = mac_costs(model, estimator)[0]
    chosen, spent = choose("activity", model, estimator, seq, states, 0,
                           3 * update_mac + 32, np.random.default_rng(0))
    assert chosen == [0]
    assert spent == update_mac + 32


def test_scheduler_falls_back_when_estimator_cannot_be_paid():
    model = RecurrentModel()
    seq = generate("feedback_switch", 48, 0)
    states = np.zeros((N_TICKS, 32, WIDTH))
    estimator = np.zeros(3 * 9)
    chosen, spent = choose("cbcc", model, estimator, seq, states, 0, 35,
                           np.random.default_rng(0))
    assert chosen == [] and spent == 0


def test_full_rollout_is_bounded_and_repeatable():
    model = RecurrentModel()
    seq = generate("feedback_switch", 49, 0)
    state_a, pred_a = rollout_full(model, seq)
    state_b, pred_b = rollout_full(model, seq)
    assert np.isfinite(state_a).all() and np.max(np.abs(state_a)) <= 1
    assert np.array_equal(state_a, state_b) and np.array_equal(pred_a, pred_b)


def test_readout_training_and_rollout_share_simultaneous_update_semantics():
    model = RecurrentModel()
    seq = generate("sparse_key", 50, 0)
    model.fit_readout([(50, seq)])
    state, _ = rollout_full(model, seq)
    h = np.zeros((32, WIDTH))
    proposals = np.vstack([model.step_circuit(h, seq.x[0], c) for c in range(4)])
    assert np.array_equal(state[0], proposals)
    assert np.isfinite(model.circuit_readout_norms).all()


def test_fixed_frequency_and_static_salience_are_distinct_controls():
    model = RecurrentModel()
    model.circuit_readout_norms[:] = [1.0, 4.0, 2.0, 3.0]
    seq = generate("sparse_key", 51, 0)
    states = np.ones((N_TICKS, 32, WIDTH))
    estimator = np.zeros(27)
    budget = 3000
    fixed, _ = choose("fixed_frequency", model, estimator, seq, states, 2, budget,
                      np.random.default_rng(0))
    static, _ = choose("no_influence", model, estimator, seq, states, 2, budget,
                       np.random.default_rng(0))
    assert fixed != static


def test_feedback_switch_has_no_postcue_target_sign_input():
    seq = generate("feedback_switch", 52, 0)
    lo, hi = seq.key_circuit * 8, (seq.key_circuit + 1) * 8
    for tick in range(seq.cue_tick + 1, len(seq.y)):
        if tick != seq.switch_tick:
            assert np.all(seq.x[tick, lo:hi] == 0)
    assert seq.target_onset == seq.switch_tick + 5


def test_joint_single_circuit_intervention_matches_single_label():
    model = RecurrentModel()
    seq = generate("sparse_key", 53, 0)
    model.fit_readout([(53, seq)])
    _, labels = paired_labels(model, seq)
    tick, circuit = 3, seq.key_circuit
    observed = joint_benefit(model, seq, tick, (circuit,))
    assert np.isclose(observed, labels[tick * 4 + circuit], atol=1e-12)


def test_cbcc_skips_when_every_predicted_benefit_is_negative():
    model = RecurrentModel()
    model.circuit_readout_norms[:] = 1.0
    seq = generate("sparse_key", 54, 0)
    states = np.zeros((N_TICKS, 32, WIDTH))
    estimator = np.r_[np.zeros(9), np.ones(9), np.asarray([0, 0, 0, -10, 0, 0, 0, 0, 0])]
    selected, _ = choose("cbcc", model, estimator, seq, states, 1, 2756,
                         np.random.default_rng(0))
    assert selected == []


def test_mac_costs_include_all_policy_dependent_multiply_adds():
    from cbcc_scheduler import mac_costs
    model = RecurrentModel()
    estimator = np.zeros(27)
    assert mac_costs(model, estimator) == (2624, 132)


def test_power_uses_actual_parameters_and_minimum_signflip_feasible_n():
    low = power_n(0.05, margin=0.05, target_effect=0.10, alpha=0.01, power=0.8)
    high = power_n(0.10, margin=0.05, target_effect=0.10, alpha=0.01, power=0.8)
    stringent = power_n(0.05, margin=0.05, target_effect=0.10, alpha=0.001, power=0.9)
    assert low >= 8 and high > low and stringent > low
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_training_seed_grouped_cv_model_beats_zero_output_for_both_families():
    from run_experiment import trained_models
    models, _ = trained_models()
    for model in models.values():
        assert len(model.training_cv_selected_mse) == 5
        assert len(model.training_cv_zero_mse) == 5
        assert np.mean(model.training_cv_selected_mse) < np.mean(model.training_cv_zero_mse)


def test_confirmatory_freeze_rejects_retired_protocol_version():
    import pytest
    from run_experiment import validate_freeze
    with pytest.raises(ValueError, match="version"):
        validate_freeze({"version": "CBCC-confirmatory-v1"})
