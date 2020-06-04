import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from backend.Hexplode import *
from joblib import Parallel, delayed
from backend.r_helper import *

# Initialize game
BOARD_SIZE = 5
HEX_NUM = 3 * np.square(BOARD_SIZE) - 3 * BOARD_SIZE + 1
hb = HexBoard(BOARD_SIZE)

# Start by playing (and recording) random games
init_games = 150

# Returns as [win, states, moves]
print("Start Random Games")
rand_games = Parallel(n_jobs = 64,
					  backend = 'threading',
					  verbose = 5)(delayed(get_random_moves)(BOARD_SIZE) for _ in range(init_games))


X, y = get_data_from_games(rand_games, HEX_NUM)

shuffle_ind = np.random.choice(len(X), len(X), replace = False)


print(X.shape)
print(y.shape)

# Initialize Deep Policy Network
print("Initialize Deep Policy")
model = Sequential([
	Dense(HEX_NUM, activation = 'relu', name = 'input'),
	Dense(3 * HEX_NUM, activation = 'relu', name = 'hidden1'),
	Dense(2 * HEX_NUM, activation = 'relu', name = 'hidden2'),
	Dense(HEX_NUM, activation = 'softmax', name = 'final'),
])

# Compile
optimizer = tf.optimizers.Adam(0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)

# Train on "good" random moves
model.fit(x = X[shuffle_ind], y = y[shuffle_ind], epochs = 30, batch_size = 64, verbose = 0)

# Test against random players
game = HexGame(BOARD_SIZE)
results = Parallel(n_jobs = 64,
					  backend = 'threading',
					  verbose = 5)(delayed(game.play_deep_vs_random)(model) for _ in range(100))

print("Score: " + str(np.mean(results)))
print("Score var: {0:0.3}".format(np.var(results)))

# -=-=- #

model_old = tf.keras.models.clone_model(model)

# Play Games v Self
for _ in range(25):
	rand_games = Parallel(n_jobs = 64,
						  backend = 'threading',
						  verbose = 5)(delayed(get_deep_moves)(model, model, BOARD_SIZE) for _ in range(64))

	X, y = get_data_from_games(rand_games, HEX_NUM)

	shuffle_ind = np.random.choice(len(X), len(X), replace = False)

	model_old = tf.keras.models.clone_model(model)

	model.fit(x = X[shuffle_ind], y = y[shuffle_ind], epochs = 20, batch_size = 64, verbose = 0)
	print("Self play round " + str(_) + " is done.")


# Test against random players
game = HexGame(BOARD_SIZE)
results = Parallel(n_jobs = 64,
					  backend = 'threading',
					  verbose = 5)(delayed(game.play_deep_vs_random)(model) for _ in range(100))

print("Score: " + str(np.mean(results)))
print("Score var: {0:0.3}".format(np.var(results)))