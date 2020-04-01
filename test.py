import numpy as np
import logging
import argparse
from rec.utils import normalize_matrix

logger = logging.getLogger()

def test_return(ret):
    if ret:
        print('OK')
    else:
        print('ERROR')

def ActualUserScores_test(items, attr, users, expand_items_by=2):
    from rec import ActualUserScores
    # random binary item_representation
    item_repr = np.random.randint(2, size=(attr, items))
    logger.info("Item representation:\n%s" % str(item_repr))
    s = ActualUserScores(users, item_repr)
    logger.info("Actual user scores (normalized)\n %s" % str(s.actual_scores))
    assert(s.actual_scores.all() == s.get_actual_user_scores().all())

    # expand items
    items += expand_items_by
    logger.info("Adding %d more items" % expand_items_by)
    item_repr = np.concatenate((item_repr,
                    np.random.randint(2, size=(attr, expand_items_by))), axis=1)
    logger.info(item_repr)
    assert(item_repr.shape[0] == attr)
    assert(item_repr.shape[1] == items)
    s.expand_items(item_repr)
    logger.info("Actual user scores (normalized)\n%s" % s.get_actual_user_scores())
    assert(s.get_actual_user_scores().all() == s.actual_scores.all())
    return True


def SIR_test(items, attr, users, item_repr=None, user_repr=None,
                        bad_user_repr=None):
    from rec import SIRModel
    def assert_correct_num_users(users, model):
        assert(users == model.num_users)
        assert(users == model.user_profiles.shape[0])
        assert(users == model.user_profiles.shape[1])
    def assert_correct_num_items(items, model):
        print(items)
        print(model.item_attributes.shape)
        print(model.num_items)
        assert(items == model.num_items)
        assert(items == model.item_attributes.shape[1])
    def assert_correct_representation(repr, model_repr):
        assert(np.all(repr) == np.all(model_repr))

    # default init
    logger.info("Default behavior (init with no arguments)")
    s = SIRModel()
    logger.info("Random item attributes\n %s" % str(s.item_attributes))
    logger.info("Random user profiles\n %s\n" % str(s.user_profiles))
    assert_correct_num_users(s.num_users, s)
    assert_correct_num_items(s.num_items, s)

    # init with given arguments
    logger.info("Init with all arguments given")
    s = SIRModel(num_users=users, num_items=items)
    logger.info("Given item attributes\n %s" % str(s.item_attributes))
    logger.info("Given user profiles\n %s\n" % str(s.user_profiles))
    assert_correct_num_users(users, s)
    assert_correct_num_items(items, s)

    # init with partially given arguments
    logger.info("Init with partially given arguments")
    s = SIRModel(num_users=users)
    assert_correct_num_users(users, s)
    assert_correct_num_items(s.num_items, s)
    logger.info("Successfully initialized with %d users" % users)
    s = SIRModel(num_items=items)
    assert_correct_num_users(s.num_users, s)
    assert_correct_num_items(items, s)
    logger.info("Successfully initialized with %d items" % items)

    # Skipping random combinations since we only have two parameters -- items and users

    logger.info("Now trying with item and/or user representations")
    if item_repr is None:
        item_repr = np.random.random(size=(1,items))
    if user_repr is None or user_repr.shape[0] != user_repr.shape[1]:
        user_repr = np.random.randint(2, size=(users, users))
    if bad_user_repr is None or bad_user_repr.shape[0] == bad_user_repr.shape[1]:
        # bad_user_repr should not be a square matrix
        bad_user_repr = np.random.randint(2, size=(user_repr.shape[0] + 2, user_repr.shape[1]))
    s = SIRModel(item_representation=item_repr)
    assert_correct_num_users(s.num_users, s)
    assert_correct_num_items(item_repr.shape[1], s)
    assert_correct_representation(item_repr, s.item_attributes)
    logger.info("Successfully initialized with following item representation\n %s" % str(item_repr))
    s = SIRModel(user_representation=user_repr)
    assert_correct_num_users(user_repr.shape[0], s)
    assert_correct_num_users(user_repr.shape[1], s)
    assert_correct_num_items(s.num_items, s)
    assert_correct_representation(user_repr, s.user_profiles)
    logger.info("Successfully initialized with following user representation\n %s" % str(user_repr))
    s = SIRModel(user_representation=user_repr, item_representation=item_repr)
    assert_correct_num_users(user_repr.shape[0], s)
    assert_correct_num_users(user_repr.shape[1], s)
    assert_correct_num_items(item_repr.shape[1], s)
    assert_correct_representation(user_repr, s.user_profiles)
    assert_correct_representation(item_repr, s.item_attributes)
    logger.info("Successfully initialized with following user representation\n %s\n \
            \nand item representation\n%s\n" % (str(user_repr), str(item_repr)))
    
    logger.info("Trying with non-matching representations. This should *NOT* succeed...")
    try:
        s = SIRModel(user_representation=bad_user_repr, item_representation=item_repr)
        assert_correct_num_users(bad_user_repr.shape[0], s)
        assert_correct_num_users(bad_user_repr.shape[1], s)
        assert_correct_num_items(item_repr.shape[1], s)
        assert_correct_representation(bad_user_repr, s.user_profiles)
        assert_correct_representation(item_repr, s.item_attributes)
    except Exception as E:
        logger.info("Failed, indeed, with: %s\n" % str(E))
    else:
        logger.info("ERROR: Did not fail")
        raise Exception("Violated rule with item and user representation!")

    logger.info("Trying default init with other params")
    num_items_per_iter = np.random.randint(5, 100)
    num_new_items = np.random.randint(20, 400)
    s = SIRModel(verbose=False,
        num_items_per_iter=num_items_per_iter,
        num_new_items=num_new_items)
    assert(num_items_per_iter == s.num_items_per_iter)
    assert(num_new_items == s.num_new_items)
    logger.info("Successfully initialized with %d new items and %d items" % (num_new_items,
        num_items_per_iter) + " per iter\n" )

    logger.info("Try letting a random user follow another...")
    user1, user2 = 0, 0
    while(user1 == user2):
        user1 = np.random.randint(s.num_users)
        user2 = np.random.randint(s.num_users)
    s.follow(user1, user2)
    assert(s.user_profiles[user1, user2])
    logger.info("User %d successfully followed user %d" % (user1, user2))

    logger.info("Try again and this should not change")
    s.follow(user1, user2)
    assert(s.user_profiles[user1, user2])
    logger.info("User %d is indeed still following user %d" % (user1, user2))

    logger.info("User %d will now unfollow user %d" % (user1, user2))
    s.unfollow(user1, user2)
    assert(s.user_profiles[user1, user2] == 0)
    logger.info("User %d successfully unfollowed user %d" % (user1, user2))

    logger.info("Try again and this should not change")
    s.unfollow(user1, user2)
    assert(s.user_profiles[user1, user2] == 0)
    logger.info("User %d is indeed still *not* following user %d" % (user1, user2))

    logger.info("Now the same users will follow each other...")
    s.add_friends(user1, user2)
    assert(s.user_profiles[user1, user2])
    assert(s.user_profiles[user2, user1])
    logger.info("Users %d and %d follow each other now" % (user1, user2))

    logger.info("Try again and this should not change")
    s.add_friends(user2, user1)
    assert(s.user_profiles[user1, user2])
    assert(s.user_profiles[user2, user1])
    logger.info("Users %d and %d are still friends" % (user1, user2))

    logger.info("Run a few steps with SocialFiltering...")
    initial_infection_state = np.copy(s.infection_state)
    s.run()
    #assert(np.all(s.infection_state) != np.all(initial_infection_state))
    logger.info("Ran default num of times successfully")

    logger.info("Now the two users will stop following each other...")
    s.remove_friends(user1, user2)
    assert(s.user_profiles[user1, user2] == 0)
    assert(s.user_profiles[user2, user1] == 0)

    logger.info("Try again and this should not change")
    s.remove_friends(user1, user2)
    assert(s.user_profiles[user1, user2] == 0)
    assert(s.user_profiles[user2, user1] == 0)
    logger.info("Users %d and %d are still *not* friends" % (user1, user2))
    
    return True


def SocialFiltering_test(items, attr, users, item_repr=None, user_repr=None,
                        bad_item_repr=None):
    from rec import SocialFiltering
    def assert_correct_num_users(users, model):
        assert(users == model.num_users)
        assert(users == model.user_profiles.shape[0])
        assert(users == model.user_profiles.shape[1])
    def assert_correct_num_items(items, model):
        assert(items == model.num_items)
        assert(items == model.item_attributes.shape[1])
    def assert_correct_item_attributes(model):
        assert(model.item_attributes.shape[0] == model.user_profiles.shape[0])
    def assert_correct_representation(repr, model_repr):
        assert(np.all(repr) == np.all(model_repr))

    # default init
    logger.info("Default behavior (init with no arguments)")
    s = SocialFiltering()
    logger.info("Random item attributes\n %s" % str(s.item_attributes))
    logger.info("Random user profiles\n %s\n" % str(s.user_profiles))
    assert_correct_num_users(s.num_users, s)
    assert_correct_num_items(s.num_items, s)
    assert_correct_item_attributes(s)

    # init with given arguments
    logger.info("Init with all arguments given")
    s = SocialFiltering(num_users=users, num_items=items)
    logger.info("Given item attributes\n %s" % str(s.item_attributes))
    logger.info("Given user profiles\n %s\n" % str(s.user_profiles))
    assert_correct_num_users(users, s)
    assert_correct_num_items(items, s)
    assert_correct_item_attributes(s)

    # init with partially given arguments
    logger.info("Init with partially given arguments")
    s = SocialFiltering(num_users=users)
    assert_correct_num_users(users, s)
    assert_correct_num_items(s.num_items, s)
    assert_correct_item_attributes(s)
    logger.info("Successfully initialized with %d users" % users)
    s = SocialFiltering(num_items=items)
    assert_correct_num_users(s.num_users, s)
    assert_correct_num_items(items, s)
    assert_correct_item_attributes(s)
    logger.info("Successfully initialized with %d items" % items)

    # Skipping random combinations since we only have two parameters -- items and users

    logger.info("Now trying with item and/or user representations")
    if item_repr is None:
        item_repr = np.zeros((3,2))
    if user_repr is None:
        user_repr = np.random.randint(2, size=(item_repr.shape[0], item_repr.shape[0]))
    if bad_item_repr is None or bad_item_repr.shape[0] == user_repr.shape[0]:
        # |U| shouldn't match user_repr.shape[0]
        bad_item_repr = np.random.randint(0, 10, size=(user_repr.shape[0] + 2, item_repr.shape[1]))
    s = SocialFiltering(item_representation=item_repr)
    assert_correct_num_users(s.num_users, s)
    assert_correct_num_items(item_repr.shape[1], s)
    assert_correct_item_attributes(s)
    assert_correct_representation(item_repr, s.item_attributes)
    logger.info("Successfully initialized with following item representation\n %s" % str(item_repr))
    s = SocialFiltering(user_representation=user_repr)
    assert_correct_num_users(user_repr.shape[0], s)
    assert_correct_num_items(s.num_items, s)
    assert_correct_item_attributes(s)
    assert_correct_representation(user_repr, s.user_profiles)
    logger.info("Successfully initialized with following user representation\n %s" % str(user_repr))
    s = SocialFiltering(user_representation=user_repr, item_representation=item_repr)
    assert_correct_num_users(user_repr.shape[0], s)
    assert_correct_num_items(item_repr.shape[1], s)
    assert_correct_item_attributes(s)
    assert_correct_representation(user_repr, s.user_profiles)
    assert_correct_representation(item_repr, s.item_attributes)
    logger.info("Successfully initialized with following user representation\n %s\n \
            \nand item representation\n%s\n" % (str(user_repr), str(item_repr)))
    
    logger.info("Trying with non-matching representations. This should *NOT* succeed...")
    try:
        s = SocialFiltering(user_representation=user_repr, item_representation=bad_item_repr)
        assert_correct_num_users(user_repr.shape[0], s)
        assert_correct_num_items(bad_item_repr.shape[1], s)
        assert_correct_item_attributes(s)
        assert_correct_representation(user_repr, s.user_profiles)
        assert_correct_representation(bad_item_repr, s.item_attributes)
    except Exception as E:
        logger.info("Failed, indeed, with: %s\n" % str(E))
    else:
        logger.info("ERROR: Did not fail")
        raise Exception("Violated rule with item and user representation!")

    logger.info("Trying default init with other params")
    num_items_per_iter = np.random.randint(5, 100)
    num_new_items = np.random.randint(20, 400)
    s = SocialFiltering(verbose=False,
        num_items_per_iter=num_items_per_iter,
        num_new_items=num_new_items)
    assert(num_items_per_iter == s.num_items_per_iter)
    assert(num_new_items == s.num_new_items)
    logger.info("Successfully initialized with %d new items and %d items" % (num_new_items,
        num_items_per_iter) + " per iter\n" )

    logger.info("Try letting a random user follow another...")
    user1, user2 = 0, 0
    while(user1 == user2):
        user1 = np.random.randint(s.num_users)
        user2 = np.random.randint(s.num_users)
    s.follow(user1, user2)
    assert(s.user_profiles[user1, user2])
    logger.info("User %d successfully followed user %d" % (user1, user2))

    logger.info("Try again and this should not change")
    s.follow(user1, user2)
    assert(s.user_profiles[user1, user2])
    logger.info("User %d is indeed still following user %d" % (user1, user2))

    logger.info("User %d will now unfollow user %d" % (user1, user2))
    s.unfollow(user1, user2)
    assert(s.user_profiles[user1, user2] == 0)
    logger.info("User %d successfully unfollowed user %d" % (user1, user2))

    logger.info("Try again and this should not change")
    s.unfollow(user1, user2)
    assert(s.user_profiles[user1, user2] == 0)
    logger.info("User %d is indeed still *not* following user %d" % (user1, user2))

    logger.info("Now the same users will follow each other...")
    s.add_friends(user1, user2)
    assert(s.user_profiles[user1, user2])
    assert(s.user_profiles[user2, user1])
    logger.info("Users %d and %d follow each other now" % (user1, user2))

    logger.info("Try again and this should not change")
    s.add_friends(user2, user1)
    assert(s.user_profiles[user1, user2])
    assert(s.user_profiles[user2, user1])
    logger.info("Users %d and %d are still friends" % (user1, user2))

    logger.info("Run a few steps with SocialFiltering...")
    s.run()
    assert(s.item_attributes[user1, :].any() != 0)
    assert(s.item_attributes[user2, :].any() != 0)
    logger.info("Ran default num of times successfully")

    logger.info("Now the two users will stop following each other...")
    s.remove_friends(user1, user2)
    assert(s.user_profiles[user1, user2] == 0)
    assert(s.user_profiles[user2, user1] == 0)

    logger.info("Try again and this should not change")
    s.remove_friends(user1, user2)
    assert(s.user_profiles[user1, user2] == 0)
    assert(s.user_profiles[user2, user1] == 0)
    logger.info("Users %d and %d are still *not* friends" % (user1, user2))
    
    return True


def ContentFiltering_test(items, attr, users, item_repr=None, user_repr=None,
                            bad_user_repr=None):
    from rec import ContentFiltering
    def assert_correct_num_users(users, model):
        assert(users == model.num_users)
        assert(users == model.user_profiles.shape[0])
    def assert_correct_num_items(items, model):
        assert(items == model.num_items)
        assert(items == model.item_attributes.shape[1])
    def assert_correct_num_attr(attr, model):
        assert(attr == model.item_attributes.shape[0] == model.user_profiles.shape[1])
    def assert_correct_representation(repr, model_repr):
        assert(np.all(repr) == np.all(model_repr))

    # default init
    logger.info("Default behavior (init with no arguments)")
    c = ContentFiltering()
    logger.info("Random item attributes\n %s" % str(c.item_attributes))
    logger.info("Random user profiles\n %s\n" % str(c.user_profiles))
    assert_correct_num_users(c.num_users, c)
    assert_correct_num_items(c.num_items, c)
    assert_correct_num_attr(c.item_attributes.shape[0], c)

    # init with given arguments
    logger.info("Init with all arguments given")
    c = ContentFiltering(num_users=users, num_items=items, num_attributes=attr)
    logger.info("Given item attributes\n %s" % str(c.item_attributes))
    logger.info("Given user profiles\n %s\n" % str(c.user_profiles))
    assert_correct_num_users(users, c)
    assert_correct_num_items(items, c)
    assert_correct_num_attr(attr, c)

    # init with partially given arguments
    logger.info("Init with partially given arguments")
    c = ContentFiltering(num_users=users)
    assert_correct_num_users(users, c)
    assert_correct_num_items(c.num_items, c)
    assert_correct_num_attr(c.item_attributes.shape[0], c)
    logger.info("Successfully initialized with %d users" % users)
    c = ContentFiltering(num_items=items)
    assert_correct_num_users(c.num_users, c)
    assert_correct_num_items(items, c)
    assert_correct_num_attr(c.item_attributes.shape[0], c)
    logger.info("Successfully initialized with %d items" % items)
    c = ContentFiltering(num_attributes=attr)
    assert_correct_num_users(c.num_users, c)
    assert_correct_num_items(c.num_items, c)
    assert_correct_num_attr(attr, c)
    logger.info("Successfully initialized with %d attributes\n" % attr)

    logger.info("Now attempting combinations of random and given arguments")
    c = ContentFiltering(num_users=users, num_items=items)
    assert_correct_num_users(users, c)
    assert_correct_num_items(items, c)
    assert_correct_num_attr(c.item_attributes.shape[0], c)
    logger.info("Successfully initialized with %d users and %d items" % (users, items))
    c = ContentFiltering(num_users=users, num_attributes=attr)
    assert_correct_num_users(users, c)
    assert_correct_num_items(c.num_items, c)
    assert_correct_num_attr(attr, c)
    logger.info("Successfully initialized with %d users and %d attributes" % (users, attr))
    c = ContentFiltering(num_attributes=attr, num_items=items)
    assert_correct_num_users(c.num_users, c)
    assert_correct_num_items(items, c)
    assert_correct_num_attr(attr, c)
    logger.info("Successfully initialized with %d items and %d attributes\n" % (items, attr))

    logger.info("Now trying with item and/or user representations")
    if item_repr is None:
        item_repr = np.array([[0,1,0], [1, 0, 0], [0, 0, 1], [1, 1, 0]])
    if user_repr is None:
        user_repr = np.random.randint(0, 10, size=(users, item_repr.shape[0]))
    if bad_user_repr is None or bad_user_repr.shape[1] == item_repr.shape[0]:
        # |A| shouldn't match item_repr.shape[0]
        bad_user_repr = np.random.randint(0, 10, size=(item_repr.shape[0], user_repr.shape[1] + 2))
    c = ContentFiltering(item_representation=item_repr)
    assert_correct_num_users(c.num_users, c)
    assert_correct_num_items(item_repr.shape[1], c)
    assert_correct_num_attr(item_repr.shape[0], c)
    assert_correct_representation(item_repr, c.item_attributes)
    logger.info("Successfully initialized with following item representation\n %s" % str(item_repr))
    c = ContentFiltering(user_representation=user_repr)
    assert_correct_num_users(user_repr.shape[0], c)
    assert_correct_num_items(c.num_items, c)
    assert_correct_num_attr(user_repr.shape[1], c)
    assert_correct_representation(user_repr, c.user_profiles)
    logger.info("Successfully initialized with following user representation\n %s" % str(user_repr))
    c = ContentFiltering(user_representation=user_repr, item_representation=item_repr)
    assert_correct_num_users(user_repr.shape[0], c)
    assert_correct_num_items(item_repr.shape[1], c)
    assert_correct_num_attr(user_repr.shape[1], c)
    assert_correct_num_attr(item_repr.shape[0], c)
    assert_correct_representation(user_repr, c.user_profiles)
    assert_correct_representation(item_repr, c.item_attributes)
    logger.info("Successfully initialized with following user representation\n %s\n \
            \nand item representation\n%s\n" % (str(user_repr), str(item_repr)))
    
    logger.info("Trying with non-matching representations. This should *NOT* succeed...")
    try:
        c = ContentFiltering(user_representation=bad_user_repr, item_representation=item_repr)
        assert_correct_num_users(bad_user_repr.shape[0], c)
        assert_correct_num_items(item_repr.shape[1], c)
        assert_correct_num_attr(bad_user_repr.shape[1], c)
        assert_correct_num_attr(item_repr.shape[0], c)
        assert_correct_representation(bad_user_repr, c.user_profiles)
        assert_correct_representation(item_repr, c.item_attributes)
    except Exception as E:
        logger.info("Failed, indeed, with: %s\n" % str(E))
    else:
        logger.info("ERROR: Did not fail")
        raise Exception("Violated rule with item and user representation!")

    logger.info("Trying default init with other params")
    num_items_per_iter = np.random.randint(5, 100)
    num_new_items = np.random.randint(20, 400)
    c = ContentFiltering(verbose=False,
        num_items_per_iter=num_items_per_iter,
        num_new_items=num_new_items)
    assert(num_items_per_iter == c.num_items_per_iter)
    assert(num_new_items == c.num_new_items)
    logger.info("Successfully initialized with %d new items and %d items" % (num_new_items,
        num_items_per_iter) + " per iter\n" )

    logger.info("Run a few steps with ContentFiltering...")
    c.run()
    assert(c.user_profiles.all() != 0)
    logger.info("Ran default num of times successfully")
    return True

# --------------------------------------------------------- #
# --------------------------------------------------------- #

if __name__ == '__main__':
    items = 3
    attr = 5
    users = 6
    logging.basicConfig(level=logging.INFO)

    # Define modules and how they map to test functions
    choices = ['content', 'user_scores', 'social', 'sir']
    choice_mapping = {'content': ContentFiltering_test,
                        'user_scores': ActualUserScores_test,
                        'social': SocialFiltering_test,
                        'sir': SIR_test}

    # Initialize parser and parse arguments
    parser = argparse.ArgumentParser(description='Test/debug recsys')
    parser.add_argument('--debug', '-d', choices=choices, required=True,
                        action='store', help='Decide on module to debug',
                        nargs='+')

    args = parser.parse_args()

    # For each module to test
    for module_name in args.debug:
        logger.info("# ------------------ #")
        logger.info('TESTING %s' % str(module_name.upper()))
        logger.info('# ------------------ #\n')
        try:
            # Use argument to find and run test function
            ret = choice_mapping[module_name](items, attr, users)
        except Exception as E:
            test_return(ret)
            raise E
        else:
            test_return(ret)

