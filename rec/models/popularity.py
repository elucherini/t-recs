import numpy as np
from rec.models import BaseRecommender
from rec.metrics import MSEMeasurement
from rec.utils import get_first_valid, is_array_valid_or_none, is_equal_dim_or_none, all_none, is_valid_or_none

class PopularityRecommender(BaseRecommender):
    def __init__(self, num_users=100, num_items=1250,
                 item_representation=None, user_representation=None,
                 actual_user_representation=None,seed=None,
                verbose=False, num_items_per_iter=10, num_new_items=30):

        if all_none(item_representation, num_items):
                raise ValueError("num_items and item_representation can't be all None")
        if all_none(user_representation, num_users):
                raise ValueError("num_users and user_representation can't be all None")

        if not is_array_valid_or_none(item_representation, ndim=2):
            raise ValueError("item_representation is not valid")
        if not is_array_valid_or_none(user_representation, ndim=2):
            raise ValueError("item_representation is not valid")

        num_items = get_first_valid(getattr(item_representation, 'shape',
                                            [None, None])[1],
                                    num_items)

        num_users = get_first_valid(getattr(user_representation, 'shape', [None])[0],
                                    num_users)

        if item_representation is None:
            item_representation = np.zeros((1, num_items), dtype=int)
        if user_representation is None:
            user_representation = np.ones((num_users, 1), dtype=int)


        if not is_equal_dim_or_none(getattr(user_representation,
                                            'shape', [None, None])[1],
                                    getattr(item_representation,
                                            'shape', [None])[0]):
            raise ValueError("user_representation.shape[1] should be the same as " + \
                             "item_representation.shape[0]")
        if not is_equal_dim_or_none(getattr(user_representation,
                                            'shape', [None])[0],
                                    num_users):
            raise ValueError("user_representation.shape[0] should be the same as " + \
                             "num_users")
        if not is_equal_dim_or_none(getattr(item_representation,
                                            'shape', [None, None])[1],
                                    num_items):
            raise ValueError("item_representation.shape[1] should be the same as " + \
                             "num_items")

        measurements = [MSEMeasurement()]

        super().__init__(user_representation, item_representation,
                         actual_user_representation, num_users, num_items,
                         num_items_per_iter, num_new_items,
                         measurements=measurements, verbose=verbose, seed=seed)

    def _update_user_profiles(self, interactions):
        histogram = np.zeros(self.num_items)
        np.add.at(histogram, interactions, 1)
        self.item_attributes[:,:] = np.add(self.item_attributes, histogram)
