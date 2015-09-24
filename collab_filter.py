import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
errors = []
for i in range(iterations):
    U = np.linalg.solve(np.dot(V.T, V) + lambda_ * np.eye(rank), np.dot(V.T, Y.T)).T
    V = np.linalg.solve(np.dot(U.T, U) + lambda_ * np.eye(rank), np.dot(U.T, Y))
    if i % 100 == 0:
        print('{}th iteration is completed'.format(i))
    errors.append(get_error(Y, U, V, valid))
Y_hat = np.dot(U, V)
print('Error of rated movies: {}'.format(get_error(Y, U, V, valid)))
"""


def get_naive_error(Y, valid):
	mean = Y[Y>0].mean()
	return np.sum((valid * (Y - mean))**2)

def get_error(Y, U, V, valid):
    return np.sum((valid * (Y - np.dot(U, V.T)))**2)

def extract_ratings(max_users, ratings_per_user):
	select_indices = []
	ratings = pd.read_csv('data/ratings.csv')
	g = ratings.groupby('userId')
	indices = g.indices
	for i in range(1,max_users+1):
		select = min(ratings_per_user, len(indices[i]))
		select_indices.extend(np.random.choice(indices[i], select, replace=False))
	select_indices = np.array(sorted(select_indices))
	return ratings.ix[select_indices]
#ratings_small = extract_ratings(100,100)
#ratings_small.to_csv('data/ratings_small.csv', index=False)

"""Take ratings of top users from most-rated movies."""
def extract_ratings2(max_users, max_movies):
	select_indices = []
	ratings = pd.read_csv('data/ratings.csv')

	g_user = ratings.groupby('userId')
	user_size = g_user.size()
	user_size.sort()
	user_size = user_size.iloc[::-1][:max_users]
	users = np.array(sorted(user_size.index))

	g_movie = ratings.groupby('movieId')
	movie_size = g_movie.size()
	movie_size.sort()
	movie_size = movie_size.iloc[::-1][:max_movies]
	movies = np.array(sorted(movie_size.index))

	res = ratings.loc[(ratings['userId'].isin(users))&(ratings['movieId'].isin(movies))]
	return res

ratings_small = extract_ratings2(100, 1000)
ratings_small.to_csv('data/ratings_small.csv', index=False)

def split_data(ratings):
	size = len(ratings)
	training_size = int(0.5 * size)
	validation_size = int(0.3 * size)
	test_size = size - training_size - validation_size

	training_indices = np.random.choice(range(size), training_size, replace=False)
	remaining = np.setdiff1d(range(size), training_indices)
	validation_indices = np.random.choice(remaining, validation_size, replace=False)
	test_indices = np.setdiff1d(remaining, validation_indices)

	return ratings.ix[training_indices], ratings.ix[validation_indices], ratings.ix[test_indices]

def solve_iter(U, V, Y, valid, rank):
    for row_idx, row in enumerate(valid):
        U[row_idx] = np.linalg.solve(np.dot(V.T, np.dot(np.diag(row), V)) + lambda_ * np.eye(rank),
                               np.dot(V.T, np.dot(np.diag(row), Y[row_idx].T))).T
    for col_idx, col in enumerate(valid.T):
        V[col_idx] = np.linalg.solve(np.dot(U.T, np.dot(np.diag(col), U)) + lambda_ * np.eye(rank),
                                 np.dot(U.T, np.dot(np.diag(col), Y[:, col_idx])))
    return U, V

######

tags = pd.read_csv('data/tags.csv')
ratings = pd.read_csv('data/ratings_small.csv')
movies = pd.read_csv('data/movies.csv')
movie_titles = movies.title.tolist()

NUM_MOVIES = len(ratings.movieId.unique())
NUM_USERS = len(ratings.userId.unique())
NUM_RATINGS = len(ratings)

training, validation, testing = split_data(ratings)
# make sure movies or users didn't disappear
assert(len(training.movieId.unique()) == len(validation.movieId.unique()) == \
	len(testing.movieId.unique()) == len(ratings.movieId.unique()))
assert(len(training.userId.unique()) == len(validation.userId.unique()) == \
	len(testing.userId.unique()) == len(ratings.userId.unique()))

matrix = training.pivot(index='userId', columns='movieId', values='rating')
Y = matrix.fillna(0).values
valid = np.isfinite(matrix).astype(np.float64).values

matrix_test = testing.pivot(index='userId', columns='movieId', values='rating')
Y_test = matrix_test.fillna(0).values
valid_test = np.isfinite(matrix_test).astype(np.float64).values

lambda_ = 0.1
rank = 2
m, n = Y.shape
iterations = 10

U = 5 * np.random.rand(m, rank)
V = 5 * np.random.rand(n, rank)

mse = [get_naive_error(Y, valid) / len(training)]
mse_test = [get_naive_error(Y_test, valid_test) / len(testing)]
for i in range(iterations):
    print('Iteration #{}'.format(i+1))
    U, V = solve_iter(U, V, Y, valid, rank)
    mse_error = get_error(Y, U, V, valid) / len(training)
    mse_test_error = get_error(Y_test, U, V, valid_test) / len(testing)
    print mse_error, mse_test_error
    mse.append(mse_error)
    mse_test.append(mse_test_error)
Y_hat = np.dot(U, V.T)


