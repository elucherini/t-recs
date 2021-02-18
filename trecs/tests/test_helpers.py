import numpy as np

import trecs.matrix_ops as mo
from trecs.base import Component


def assert_equal_arrays(a, b):
    a, b = mo.extract_values(a, b)
    assert np.array_equal(a, b)


def assert_correct_num_users(num_users, model, size):
    assert num_users == model.num_users
    assert num_users == size


def assert_correct_num_items(num_items, model, size):
    assert num_items == model.num_items
    assert num_items == size


def assert_correct_size_generic(attribute, num_attribute, size):
    assert num_attribute == attribute
    assert num_attribute == size


def assert_correct_representation(repr, model_repr):
    model_repr = mo.extract_value(model_repr)
    assert np.array_equal(repr, model_repr)


def assert_social_graph_following(graph, user, following):
    assert graph[following, user]


def assert_social_graph_not_following(graph, user, not_following):
    assert graph[not_following, user] == 0


def assert_not_none(repr):
    assert repr is not None


def assert_equal_measurements(meas1, meas2):
    for key, val in meas1.items():
        assert key in meas2
        assert_equal_arrays(val, meas2[key])
    for key, val in meas2.items():
        assert key in meas1
        assert_equal_arrays(val, meas1[key])


def assert_equal_system_state(systate1, systate2):
    # check identical keys
    assert_equal_arrays(systate1.keys(), systate2.keys())
    for key, val in systate1.items():
        assert key in systate2
        if isinstance(val, list):
            for i, item in enumerate(val):
                assert_equal_arrays(item, systate2[key][i])
        elif val is None:
            assert val == systate2[key] == None
        elif isinstance(val, np.ndarray):
            assert_equal_arrays(val, systate2[key])
        elif isinstance(val, Component):
            assert_equal_arrays(val.value, systate2[key].value)
