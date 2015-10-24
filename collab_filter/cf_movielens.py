import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

movie_dir = '../movie_data'

def als_naive(U, V, Y, valid, lambda_, iterations):
    errors = []
    for i in range(iterations):
        U = np.linalg.solve(np.dot(V.T, V) + lambda_ * np.eye(rank), np.dot(V.T, Y.T)).T
        V = np.linalg.solve(np.dot(U.T, U) + lambda_ * np.eye(rank), np.dot(U.T, Y))
        if i % 100 == 0:
            print('{}th iteration is completed'.format(i))
        errors.append(get_error(Y, U, V, valid))
    Y_hat = np.dot(U, V)
    print('Error of rated movies: {}'.format(get_error(Y, U, V, valid)))


def get_naive_error(Y, valid):
    n = valid.sum()
    mean = Y[Y>0].mean()
    return np.sqrt(np.sum((valid * (Y - mean))**2) / n)

def get_error_rmse(Y, U, V, valid):
    n = valid.sum()
    return np.sqrt(np.sum((valid * (Y - np.dot(U, V.T)))**2) / n)

"""Extract random ratings from users."""
def extract_ratings(max_users, ratings_per_user):
    select_indices = []
    ratings = pd.read_csv('{}/ratings.csv'.format(movie_dir))
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
    ratings = pd.read_csv('{}/ratings.csv'.format(movie_dir))

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
#ratings_small = extract_ratings2(100, 1000)
#ratings_small.to_csv('data/ratings_small.csv', index=False)

def split_data(ratings):
    size = len(ratings)
    training_size = int(0.5 * size)
    validation_size = int(0.3 * size)
    test_size = size - training_size - validation_size

    training_indices = np.random.choice(range(size), training_size, replace=False)
    remaining = np.setdiff1d(range(size), training_indices)
    validation_indices = np.random.choice(remaining, validation_size, replace=False)
    test_indices = np.setdiff1d(remaining, validation_indices)

    if type(ratings) == pd.DataFrame:
        return ratings.ix[training_indices], ratings.ix[validation_indices], ratings.ix[test_indices]
    return ratings[training_indices], ratings[validation_indices], ratings[test_indices]


def solve_iter(U, V, Y, valid, lambda_, rank_):
    for row_idx, row in enumerate(valid):
        U[row_idx] = np.linalg.solve(np.dot(V.T, np.dot(np.diag(row), V)) + lambda_ * np.eye(rank_),
                               np.dot(V.T, np.dot(np.diag(row), Y[row_idx].T))).T
    for col_idx, col in enumerate(valid.T):
        V[col_idx] = np.linalg.solve(np.dot(U.T, np.dot(np.diag(col), U)) + lambda_ * np.eye(rank_),
                                 np.dot(U.T, np.dot(np.diag(col), Y[:, col_idx])))
    return U, V

def validate(matrix):
    if type(matrix) == pd.DataFrame:
        return np.isfinite(matrix).astype(np.float64).values
    return np.isfinite(matrix).astype(np.float64)

######

def gen_genre_matrix(movies, genres):
    arr = []
    for genre in genres:
        arr.append((movies.genres.str.find(genre)>=0).astype('int').values)
    return np.array(arr).T

def cf_movies(): 
    tags = pd.read_csv('{}/tags.csv'.format(movie_dir))
    ratings = pd.read_csv('{}/ratings_small.csv'.format(movie_dir))
    movies = pd.read_csv('{}/movies.csv'.format(movie_dir))
    movie_titles = movies.title.tolist()
    genres = np.unique(np.array(list(itertools.chain.from_iterable([i.split('|') for i in movies.genres.tolist()]))))

    genre_matrix = gen_genre_matrix(movies, genres)
    # also maybe calc for each user, the avg rating for each genre

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
    valid = validate(matrix)

    matrix_validate = validation.pivot(index='userId', columns='movieId', values='rating')
    Y_validate = matrix_validate.fillna(0).values
    valid_validate = validate(matrix_validate)

    matrix_test = testing.pivot(index='userId', columns='movieId', values='rating')
    Y_test = matrix_test.fillna(0).values
    valid_test = validate(matrix_test)

    return Y, valid, Y_validate, valid_validate, Y_test, valid_test


def collab_filter(Y, valid, Y_validate, valid_validate, lambda_, rank_, iterations):
    d = {}
    m, n = Y.shape
    print "Lambda: {}, rank: {}".format(lambda_, rank_)
    U = 5 * np.random.rand(m, rank_)
    V = 5 * np.random.rand(n, rank_)

    rmse = [get_naive_error(Y, valid)]
    rmse_validate = [get_naive_error(Y_validate, valid_validate)]
    for i in range(iterations):
        U, V = solve_iter(U, V, Y, valid, lambda_, rank_)
        rmse_error = get_error_rmse(Y, U, V, valid)
        rmse_validate_error = get_error_rmse(Y_validate, U, V, valid_validate)
        #print 'Iteration #{}'.format(i+1), rmse_error, rmse_validate_error
        rmse.append(rmse_error)
        rmse_validate.append(rmse_validate_error)
    Y_hat = np.dot(U, V.T)
    d[(lambda_, rank_)] = min(rmse_validate)
    print rmse_error, rmse_validate_error
    return U, V

def cf_test(U, V, Y_test, valid_test):
    return get_error_rmse(Y_test, U, V, valid_test)


lambdas = [0.05]
ranks = [3]
iterations = 15

def run_cf_movies():
    Y, valid, Y_validate, valid_validate, Y_test, valid_test = cf_movies()
    for lambda_, rank_ in itertools.product(lambdas, ranks):
        U, V = collab_filter(Y, valid, Y_validate, valid_validate, lambda_, rank_, iterations)
    print cf_test(U, V, Y_test, valid_test)
    return U, V

