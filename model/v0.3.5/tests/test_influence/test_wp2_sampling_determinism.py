from experiments.WP2.generate_labels import selected_nodes

def test_wp2_training_node_sample_is_deterministic_without_replacement():
    a=selected_nodes('chain',64,'smooth',64001,.1,'train')
    b=selected_nodes('chain',64,'smooth',64001,.1,'train')
    assert a.tolist()==b.tolist()
    assert len(a)==16 and len(set(a.tolist()))==16
    assert not (set(a.tolist())-set(range(64)))

def test_validation_and_test_cover_all_nodes():
    assert selected_nodes('chain',64,'smooth',64011,.1,'validation').tolist()==list(range(64))
    assert selected_nodes('chain',64,'smooth',64021,.1,'test').tolist()==list(range(64))
