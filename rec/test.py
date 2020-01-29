import numpy
from rec import ActualUserScores

items = 3
attr = 5
users = 5

item_repr = np.random.randint(2, (attr, items))
s = ActualUserScores(users, item_repr)