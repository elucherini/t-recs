import numpy as np
from rec import Users, ContentFiltering, SocialFiltering, BassModel
import test_utils
import pytest

class TestUsers:
    def test_generic(self, items=10, attr=5, users=6, expand_items_by=2):
        with pytest.raises(ValueError):
            s = Users()
        with pytest.raises(TypeError):
            s = Users(actual_user_profiles='wrong type')
        with pytest.raises(TypeError):
            s = Users(actual_user_profiles=None, size='wrong type')
        with pytest.raises(TypeError):
            s = Users(size='wrong_type')
        s = Users(size=(users,attr))
        assert(s.actual_user_profiles.shape == (users, attr))
        s = Users(actual_user_profiles=np.random.randint(5, size=(users, attr)))
        assert(s.actual_user_profiles.shape == (users, attr))

    def test_content(self, items=10, attr=5, users=6, expand_items_by=2):
        """WARNING Before running this, make sure ContentFiltering is working properly"""
        # user_repr = actual_user_repr
        item_repr = np.random.randint(2, size=(attr, items))
        actual_user_repr = np.random.randint(15, size=(users, attr))
        model = ContentFiltering(user_representation=actual_user_repr,
                                 item_representation=item_repr)
        s = Users(actual_user_repr)
        s.compute_user_scores(model.train)
        test_utils.assert_equal_arrays(s.actual_user_scores,
                                       model.train(s.actual_user_profiles,
                                                   normalize=True))
        test_utils.assert_equal_arrays(s.actual_user_scores,
                                   model.predicted_scores)

        # user_repr != actual_user_repr
        user_repr =np.random.randint(15, size=(users, attr))
        model = ContentFiltering(user_representation=user_repr,
                                 item_representation=item_repr)
        assert(model.user_profiles.shape == actual_user_repr.shape)
        s = Users(actual_user_repr)
        s.compute_user_scores(model.train)
        print(np.array_equal(s.actual_user_scores, model.train(s.actual_user_profiles,
                                                              model.item_attributes,
                                                              normalize=True)))
        test_utils.assert_equal_arrays(s.actual_user_scores,
                                       model.train(s.actual_user_profiles,
                                                   normalize=True))

if __name__ == '__main__':
    t = TestUsers()
    t.test_content()
