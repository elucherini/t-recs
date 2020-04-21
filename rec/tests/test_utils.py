import numpy as np

def assert_equal_arrays(a, b):
  assert(np.all(a) == np.all(b))

def assert_correct_num_users(num_users, model, size):
    assert(num_users == model.num_users)
    assert(num_users == size)

def assert_correct_num_items(num_items, model, size):
    assert(num_items == model.num_items)
    assert(num_items == size)

def assert_correct_size_generic(attribute, num_attribute, size):
    assert(num_attribute == attribute)
    assert(num_attribute == size)

def assert_correct_representation(repr, model_repr):
    assert(np.all(repr) == np.all(model_repr))

def assert_social_graph_following(graph, user, following):
    assert(graph[following, user])

def assert_social_graph_not_following(graph, user, not_following):
    assert(graph[not_following, user] == 0)
