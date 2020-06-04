import numpy as np
from backend import game
from backend.funcs import softmax
# import multiprocessing as mp
from joblib import Parallel, delayed
# from numba import jitclass  # import the decorator
# from numba import int32, float32


class HexBoard(object):
	def __init__(self, size):
		self.size = size
		self.turnNum = 0
		self.maSize = 2 * size - 1
		self.num_hex = 3 * np.square(size) - 3 * size + 1
		self.board = np.zeros((self.maSize, self.maSize)).astype(np.int)

	def newGame(self):
		self.turnNum = 0
		self.board = np.zeros((self.maSize, self.maSize))

	def move(self, r, c):  # Put stone on board
		if self.checkWin() == 0:
			side = (self.turnNum % 2) * (-2) + 1
			if self.legal(r, c):
				if side * self.board[(r, c)] >= 0:
					self.board[r, c] += side
					self.iterate(side)
					self.turnNum += 1
					return False
				else:
					return "enemy stone"
			else:
				return "out of bounds"

	def checkWin(self):
		if self.turnNum >= 2:
			if not (self.board > 0).any():
				return -1
			if not (self.board < 0).any():
				return 1
		return 0

	def count(self, side = 1):
		score = int(np.sum(self.board[np.where(side * self.board > 0)]))
		return score

	def legal(self, r, c):  # Helper function to see if move was legal
		return self.size + r - 1 >= c >= r - self.size + 0.5 and 0 <= r < self.maSize and 0 <= c < self.maSize

	def legalco(self, coo):  # Helper function as above but with (x,y) as input
		return self.size + coo[0] - 1 >= coo[1] >= coo[0] - self.size + 0.5 and \
			   0 <= coo[0] < self.maSize and 0 <= coo[1] < self.maSize

	def iterate(self, side):  # Start exploding the tiles if neccassary
		oldBoard = np.array(self.board)
		ttexp = []  # tiles to explode

		while self.checkWin() == 0:
			# print("iter \n")
			# print(self.board)

			for r in range(self.maSize):
				for c in range(self.maSize):
					if abs(self.board[r, c]) >= len(self.touching((r, c))):
						ttexp = ttexp + [(r, c)]

			self.explode(ttexp, side)

			ttexp = []

			if (oldBoard == self.board).all():
				break

			oldBoard = np.array(self.board)

	def touching(self, coo):
		# Returns List of legal coordinates touching given
		touch = []
		if self.legalco((coo[0], coo[1] - 1)): touch = touch + [(coo[0], coo[1] - 1)]  # W
		if self.legalco((coo[0] - 1, coo[1] - 1)): touch = touch + [(coo[0] - 1, coo[1] - 1)]  # NW
		if self.legalco((coo[0] - 1, coo[1])): touch = touch + [(coo[0] - 1, coo[1])]  # NE

		if self.legalco((coo[0], coo[1] + 1)): touch = touch + [(coo[0], coo[1] + 1)]  # E
		if self.legalco((coo[0] + 1, coo[1] + 1)): touch = touch + [(coo[0] + 1, coo[1] + 1)]  # SE
		if self.legalco((coo[0] + 1, coo[1])): touch = touch + [(coo[0] + 1, coo[1])]  # SW

		return touch

	def explode(self, ttexp, side):  # distribute the stones
		for coo in ttexp:
			self.board[coo] -= side * len(self.touching(coo))
			for tou in self.touching(coo):
				self.board[tou] = side * abs(self.board[tou]) + side

	def linearBoard(self):  # Returns value of all board positions in 1D array
		linBoard = []
		for r in range(self.maSize):
			for c in range(self.maSize):
				if self.legal(r, c):
					linBoard = linBoard + [self.board[(r, c)]]
		return linBoard

	def cooFromLinSpace(self, ind):
		temp = 0
		for r in range(self.maSize):
			for c in range(self.maSize):
				if self.legal(r, c):
					if temp == ind:
						return r, c
					temp += 1

	def get_sum(self):
		return np.sum(self.board)

	def get_prob_board_from_lin(self, linboard):
		temp = 0
		square_board = np.zeros((self.maSize, self.maSize))
		for r in range(self.maSize):
			for c in range(self.maSize):
				if self.legal(r, c):
					square_board[r, c] = linboard[temp]
					temp += 1
		return square_board


class HexGame(game.Game):
	def __init__(self, board_size):
		self.bSize = board_size
		self.hexNum = 3 * board_size ** 2 - 3 * board_size + 1
		self.board = HexBoard(board_size)
		self.turn = 0

	def play(self, players):
		# score = np.zeros(len(players))
		with Parallel(n_jobs = len(players), prefer = 'threads', verbose = 1) as parallel:
			# for n, agtA in enumerate(players):
			winsA = parallel(delayed(self.play_random_enemies)(agtA, players, side = 1, num_en = 5) for agtA in players)
			winsB = parallel(
				delayed(self.play_random_enemies)(agtA, players, side = -1, num_en = 5) for agtA in players)
		return np.array(winsA) + np.array(winsB)

	def play_random_enemies(self, agtA, enemy_pool, side = 0, num_en = 5):
		enemy_agents = np.random.choice(enemy_pool, num_en, replace = False)
		if side == 1:
			wins = [self.play_once_safe(agtA, agtB)[0] for agtB in enemy_agents]
		else:
			wins = [self.play_once_safe(agtB, agtA)[1] for agtB in enemy_agents]
		return np.sum(wins)

	def play_once_safe(self, agtA, agtB):  # play with upper limit on moves
		HB = HexBoard(self.bSize)
		numTurns = 0
		while HB.checkWin() == 0 and numTurns < 150:

			# Move by A
			# Get linearized board
			linBoard = HB.linearBoard()
			# Calculate illegal moves
			invalid_moves = np.array([100. * ((p >= 0) - 1) for p in linBoard])
			# Agent Calculates move
			yHat = softmax(agtA.forward(linBoard).squeeze() + invalid_moves)
			# Get arg max
			linMove = pick_random(yHat)
			# Coordinate Transformation
			coo = HB.cooFromLinSpace(linMove)
			# Move
			HB.move(coo[0], coo[1])

			if HB.checkWin() > 0:
				return 3.5, -1.

			# Move by B
			# Get linearized board * (-1)
			linBoard = [(-1) * i for i in HB.linearBoard()]
			# Calculate illegal moves
			invalid_moves = np.array([100. * ((p >= 0) - 1) for p in linBoard])
			# Agent calculates move
			yHat = softmax(agtB.forward(linBoard).squeeze() + invalid_moves)
			# Get arg max
			linMove = pick_random(yHat)
			# Coordinate Transformation
			coo = HB.cooFromLinSpace(linMove)
			# Move
			HB.move(coo[0], coo[1])

			if HB.checkWin() < 0:
				return -1., 3.5

			numTurns += 1
		print("long game")
		return 1., 1.

	def play_self_safe(self, agt):
		HB = HexBoard(self.bSize)
		numTurns = 0
		while HB.checkWin() == 0 and numTurns < 80:
			if numTurns >= 2:
				return 1
			# Move performed by i
			yHat = agt.forward(HB.linearBoard())
			linMove = np.argmax(yHat)
			coo = HB.cooFromLinSpace(linMove)
			if HB.move(coo[0], coo[1]):
				return -2

			if HB.checkWin() > 0:
				return 0.1

			# Move performed by j
			yHat = agt.forward([(-1) * i for i in HB.linearBoard()])
			linMove = np.argmax(yHat)
			coo = HB.cooFromLinSpace(linMove)
			if HB.move(coo[0], coo[1]):
				return -2

			if HB.checkWin() < 0:
				return 0.1

			numTurns += 1
		return 1

	def play_once_print(self, agtA, agtB):  # play with upper limit on moves
		HB = HexBoard(self.bSize)
		numTurns = 0
		while HB.checkWin() == 0 and numTurns < 100:
			# Move performed by i
			linBoard = HB.linearBoard()
			invalid_moves = np.array([100. * ((p >= 0) - 1) for p in linBoard])
			yHat = softmax(agtA.forward(linBoard).squeeze() + invalid_moves)
			linMove = np.argmax(yHat)
			coo = HB.cooFromLinSpace(linMove)
			HB.move(coo[0], coo[1])

			print("Move A:")
			print(HB.board)
			print()
			if HB.checkWin() > 0:
				break

			# Move performed by j
			linBoard = [(-1) * i for i in HB.linearBoard()]  # *(-1) So the NN sees enemy as negative
			invalid_moves = np.array([100. * ((p >= 0) - 1) for p in linBoard])
			yHat = softmax(agtB.forward(linBoard).squeeze() + invalid_moves)
			linMove = np.argmax(yHat)
			coo = HB.cooFromLinSpace(linMove)
			if HB.move(coo[0], coo[1]):
				print("j lost by invalid move")
				print(coo)
				return 1
			print("Move B:")
			print(HB.board)
			print()

			if HB.checkWin() < 0:
				break

			numTurns += 1
		return HB.checkWin()

	def play_once_random(self, agtA, agtB, print_bool = True):  # play with upper limit on moves
		HB = HexBoard(self.bSize)
		numTurns = 0
		while HB.checkWin() == 0 and numTurns < 150:
			# Move performed by i
			linBoard = HB.linearBoard()
			invalid_moves = np.array([1000. * ((p >= 0) - 1) for p in linBoard])
			yHat = softmax(agtA.forward(linBoard).squeeze() + invalid_moves)
			linMove = pick_random(yHat)
			coo = HB.cooFromLinSpace(linMove)
			HB.move(coo[0], coo[1])

			if print_bool:
				print("Move A:")
				print(HB.board)
				print()
			if HB.checkWin() > 0:
				break

			# Move performed by j
			linBoard = [(-1) * i for i in HB.linearBoard()]  # *(-1) So the NN sees enemy as negative
			invalid_moves = np.array([1000. * ((p >= 0) - 1) for p in linBoard])
			yHat = softmax(agtB.forward(linBoard).squeeze() + invalid_moves)
			linMove = pick_random(yHat)
			coo = HB.cooFromLinSpace(linMove)
			if HB.move(coo[0], coo[1]):
				print("j lost by invalid move")
				print(coo)
				return 1
			if print_bool:
				print("Move B:")
				print(HB.board)
				print()

			if HB.checkWin() < 0:
				break

			numTurns += 1
		return HB.checkWin()

	def play_deep_vs_random(self, model):
		HB = HexBoard(self.bSize)
		while HB.checkWin() == 0:

			# Move performed by i
			linBoard = HB.linearBoard()

			invalid_moves = np.array([1000. * ((p >= 0) - 1) for p in linBoard])
			yHat = softmax(model(np.array(linBoard).reshape((1, HB.num_hex))) + invalid_moves).squeeze()
			linMove = pick_random(yHat)

			coo = HB.cooFromLinSpace(linMove)
			HB.move(coo[0], coo[1])

			if HB.checkWin() > 0:
				break

			# Move performed by j
			linBoard = [(-1) * i for i in HB.linearBoard()]  # *(-1) So the NN sees enemy as negative

			invalid_moves = np.array([1000. * ((p >= 0) - 1) for p in linBoard])
			yHat = softmax(np.ones(HB.num_hex) + invalid_moves)
			linMove = pick_random(yHat)

			coo = HB.cooFromLinSpace(linMove)
			HB.move(coo[0], coo[1])

			if HB.checkWin() < 0:
				break

		return HB.checkWin()


def get_random_moves(board_size = 5):
	HB = HexBoard(board_size)
	numTurns = 0
	moves = []
	states = []
	while HB.checkWin() == 0:
		# Move performed by i
		linBoard = HB.linearBoard()
		states.append(linBoard)

		invalid_moves = np.array([1000. * ((p >= 0) - 1) for p in linBoard])
		yHat = softmax(np.ones(HB.num_hex) + invalid_moves)
		linMove = pick_random(yHat)
		moves.append(linMove)

		coo = HB.cooFromLinSpace(linMove)
		HB.move(coo[0], coo[1])

		if HB.checkWin() > 0:
			break

		# Move performed by j
		linBoard = [(-1) * i for i in HB.linearBoard()]  # *(-1) So the NN sees enemy as negative
		states.append(linBoard)

		invalid_moves = np.array([1000. * ((p >= 0) - 1) for p in linBoard])
		yHat = softmax(np.ones(HB.num_hex) + invalid_moves)
		linMove = pick_random(yHat)
		moves.append(linMove)

		coo = HB.cooFromLinSpace(linMove)
		HB.move(coo[0], coo[1])

		if HB.checkWin() < 0:
			break

		numTurns += 1

	return HB.checkWin(), states, moves


def get_deep_moves(model, model_old, board_size = 5):
	HB = HexBoard(board_size)
	numTurns = 0
	moves = []
	states = []
	while HB.checkWin() == 0:
		# Move performed by i
		linBoard = HB.linearBoard()
		states.append(linBoard)

		invalid_moves = np.array([1000. * ((p >= 0) - 1) for p in linBoard])
		yHat = softmax(model(np.array(linBoard).reshape((1, HB.num_hex))) + invalid_moves).squeeze()
		linMove = pick_random(yHat)
		moves.append(linMove)

		coo = HB.cooFromLinSpace(linMove)
		HB.move(coo[0], coo[1])

		if HB.checkWin() > 0:
			break

		# Move performed by j
		linBoard = [(-1) * i for i in HB.linearBoard()]  # *(-1) So the NN sees enemy as negative
		states.append(linBoard)

		invalid_moves = np.array([1000. * ((p >= 0) - 1) for p in linBoard])
		yHat = softmax(model_old(np.array(linBoard).reshape((1, HB.num_hex))) + invalid_moves).squeeze()
		linMove = pick_random(yHat)
		moves.append(linMove)

		coo = HB.cooFromLinSpace(linMove)
		HB.move(coo[0], coo[1])

		if HB.checkWin() < 0:
			break

		numTurns += 1

	return HB.checkWin(), states, moves


class HexIterGame():
	def __init__(self, board_size):
		self.bSize = board_size
		self.hexNum = 3 * board_size ** 2 - 3 * board_size + 1
		self.board = HexBoard(board_size)
		self.turn = 0

	def agent_move(self, agent):
		if self.turn % 2 == 0:
			linBoard = self.board.linearBoard()
		else:
			linBoard = [(-1) * p for p in self.board.linearBoard()]
		invalid_moves = np.array([100. * ((p >= 0) - 1) for p in linBoard])
		yHat = softmax(agent.forward(linBoard).squeeze() + invalid_moves)
		linMove = pick_random(yHat)
		coo = self.board.cooFromLinSpace(linMove)
		self.board.move(coo[0], coo[1])
		self.turn += 1
		return 0

	def human_move(self, move):
		coo = self.board.cooFromLinSpace(move)
		if self.board.move(coo[0], coo[1]):
			return 1
		self.turn += 1
		return 0

	def get_lin_board(self):
		return self.board.linearBoard()


def pick_random(probs):
	return np.random.choice(range(len(probs)), 1, p = probs)
