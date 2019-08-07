import argparse
import numpy as np
import constants as const
import matplotlib.pyplot as plt
from enum import Enum

from popularity import PopularityRecommender
from content import ContentFiltering
from debug import Debug

# Supported recommender systems, DO NOT CHANGE
rec_dict = {'popularity':PopularityRecommender, 'content':ContentFiltering}

# DO NOT CHANGE THE KEYS OF THE FOLLOWING DICTIONARIES
# Supported additional arguments for each recommender system
# No additional arguments are supported for popularity rec sys
rec_args = {'popularity': None,
            # A: number of attributes (must be integer);
            'content': {'A': None,
            # items_representation: representation of items based on 
            # attributes (must be matrix)
                        'items_representation': None}}
# Supported debug options, each representing a module
debug_opt = {'MEASUREMENTS': False,
            'USER_PREFERENCES': True,
            'RECOMMENDER': True}

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

if __name__ == '__main__':
    # Set up
    rec_args['content']['A'] = 5
    rec_args['content']['items_representation'] = np.zeros((const.NUM_ITEMS, 
        rec_args['content']['A']), dtype=int)
    
    for i, row in enumerate(rec_args['content']['items_representation']):
        A = rec_args['content']['A']
        n_indices = np.random.randint(1, A)
        indices = np.random.randint(A, size=(n_indices))
        row[indices] = 1
        rec_args['content']['items_representation'][i,:] = row
    rec_args['content']['items_representation'] = rec_args['content']['items_representation'].T

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('recommender', metavar='r', type=str, nargs=1,
        help='Type of recommender system', choices=set(rec_dict.keys()))
    parser.add_argument('--debug', help='Measurement debug info', action='store_true')

    # Configure and initialize debugger
    debugger = Debug(list(debug_opt.keys()), list(debug_opt.values()))

    # Create instance
    args = parser.parse_args()
    rec_type = args.recommender[0]

    if rec_args[rec_type] is None:
        rec = rec_dict[rec_type](const.NUM_USERS, const.NUM_ITEMS,
            num_items_per_iter=const.NUM_ITEMS_PER_ITER, randomize_recommended=True,
            user_preferences=True, debugger=debugger)
    else:
        rec = rec_dict[rec_type](const.NUM_USERS, const.NUM_ITEMS,
            num_items_per_iter=const.NUM_ITEMS_PER_ITER, randomize_recommended=True,
            user_preferences=True, debugger=debugger, **rec_args[rec_type])

    # Startup
    rec.startup_and_train(timesteps=const.NUM_STARTUP_ITER, measurement_visualization_rule=False)

    # Runtime
    rec.run(timesteps=const.TIMESTEPS - const.NUM_STARTUP_ITER, train_between_steps=True,
    measurement_visualization_rule="% 50 == 0")

    delta_t = rec.get_heterogeneity()
    