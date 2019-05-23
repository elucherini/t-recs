import argparse
import sys
import numpy as np
import constants as const
import matplotlib.pyplot as plt

from popularity import PopularityRecommender
from content import ContentFiltering

if __name__ == '__main__':
    # Supported recommender systems
    rec_dict = {'popularity':PopularityRecommender, 'content':ContentFiltering}
    # Supported additional arguments for each recommender system
    rec_args = {'popularity': None, 
        'content': {'A':100, 'items_representation': None}
        }
    choices = set(rec_dict.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument('recommender', metavar='r', type=str, nargs=1, help='Type of recommender system',
        choices=choices)
    parser.add_argument('--debug', '-d', help='Debug info', action='store_true')

    args = parser.parse_args()
    rec_type = args.recommender[0]

    #print(rec_args[args.recommender[0]])
    if rec_args[rec_type] is None:
        rec = rec_dict[rec_type](const.NUM_USERS, const.NUM_ITEMS, num_startup_iter=const.NUM_STARTUP_ITER,
            num_items_per_iter=const.NUM_ITEMS_PER_ITER, randomize_recommended=True, user_preference=False)
    else:
        rec = rec_dict[rec_type](const.NUM_USERS, const.NUM_ITEMS, num_startup_iter=const.NUM_STARTUP_ITER,
            num_items_per_iter=const.NUM_ITEMS_PER_ITER, randomize_recommended=True, user_preference=False, 
            **rec_args[rec_type])

    print('Num items:', const.NUM_ITEMS, '\nUsers:', const.NUM_USERS, '\nItems per iter:', const.NUM_ITEMS_PER_ITER)
    # Startup
    rec.interact_startup()
    rec.train()

    users = np.arange(const.NUM_USERS, dtype=int)

    # Runtime
    for t in range(const.TIMESTEPS - const.NUM_STARTUP_ITER):
        plot = False
        if args.debug:
            if t % 50 == 0 or t == const.TIMESTEPS - const.NUM_STARTUP_ITER - 1:
                plot=True
        #print('New', num_new_items, 'Rec', num_recommended)
        rec.interact(user_vector=users, plot=plot)
        rec.train()

    delta_t = rec.get_delta()
    plt.style.use('seaborn-whitegrid')
    plt.plot(np.arange(len(delta_t)), delta_t)
    plt.show()