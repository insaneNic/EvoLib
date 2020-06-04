import numpy as np


def get_data_from_games(rand_games, num_hex = 61):
	X = []
	y = []

	# Go through random games and make into usable data
	for game_hist in rand_games:
		ind_push = (game_hist[0] - 1) // (-2)
		num_turn = len(game_hist[1])
		for i, state_move in enumerate(zip(game_hist[1], game_hist[2])):
			if i % 2 == ind_push:
				X.append(state_move[0])
				y.append(get_linear_move_from_ind(state_move[1], num_hex))

	return np.array(X), np.array(y)


def get_linear_move_from_ind(ind, num_hex = 61):
	x = np.zeros(num_hex)
	x[ind] = 1.
	return x