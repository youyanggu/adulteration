import numpy as np

def split_line(line):
    food = line.split(',')[0].strip()
    adulterants = line.split(', ')[-1].strip().split(' ')
    adulterants = [a.split(':')[0] for a in adulterants]
    return food, adulterants

def get_index(a, food_adulterants, rank):
    if a not in rank:
        return None
    if a in food_adulterants:
        return food_adulterants.index(a)
    else:
        new_rank = [i for i in rank if i not in food_adulterants]
        return new_rank.index(a)+len(food_adulterants)

def sort_food_map(food_map, rank):
    new_food_map = {}
    for food, adulterants in food_map.iteritems():
        new_ordering = []
        for r in rank:
            if r in adulterants:
                new_ordering.append(r)
        assert set(new_ordering)==set(adulterants)
        new_food_map[food] = new_ordering
    return new_food_map

## TRAINING ##

arr = []
with open('train.dat', 'rb') as f:
    for line in f:
        arr.append(line[1:-1])
d_count = {}
food_map = {}
for line in arr:
    food, adulterants = split_line(line)
    food_map[food] = adulterants
    for a in adulterants:
        if a in d_count:
            d_count[a] += 1
        else:
            d_count[a] = 1
rank = sorted(d_count, key=d_count.get, reverse=True)
food_map = sort_food_map(food_map, rank)

entries = sum([len(i) for i in food_map.values()])
total_possibilities = len(food_map) * len(rank)
print 'Sparsity: {} / {} = {:.1%}'.format(
    entries, total_possibilities, entries*1./total_possibilities)

## TESTING ##

arr_test = []
with open('test.dat', 'rb') as f:
    for line in f:
        arr_test.append(line[1:-1])

found = not_found = 0
unknown_adulterants = 0
precisions = []
for line in arr_test:
    food, adulterants = split_line(line)
    if food in food_map:
        found += 1
    else:
        not_found += 1
    indices = []
    for a in adulterants:
        food_adulterants = food_map.get(food, [])
        index = get_index(a, food_adulterants, rank)
        if index is None:
            unknown_adulterants += 1
        else:
            indices.append(index)
    #indices = np.random.choice(len(rank), len(adulterants))
    if len(indices) == 0:
        continue
    indices = sorted(indices)
    mean_precision = 0
    for i, ind in enumerate(indices):
        mean_precision += (i+1.)/(ind+1)
    mean_precision /= len(indices)
    precisions.append(mean_precision)
precisions = np.array(precisions)
print 'Found: {}; Not found: {}; Unknown adulterants: {}'.format(
    found, not_found, unknown_adulterants)
print 'MAP:', precisions.mean()


