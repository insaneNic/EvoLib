import numpy as np
from backend import game


class HexBoard(object):
	def __init__(self, size):
		self.size = size
		self.turnNum = 0
		self.maSize = 2 * size - 1
		self.board = np.zeros((self.maSize, self.maSize))

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
		if self.checkWin() == -1:
			return "neg wins"
		return "pos wins"

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
	def __init__(self, borSize):
		self.bSize = borSize
		self.hexNum = 3 * borSize ** 2 - 3 * borSize + 1

	def play(self, players):
		score = np.zeros(len(players))
		for n, agtA in enumerate(players):
			print("\rRow " + str(n), end = '')
			for _ in range(4):
				agtB = np.random.choice(players, 1)[0]
				win = self.play_once_safe(agtA, agtB, 0)
				score[n] += win

				agtB = np.random.choice(players, 1)[0]
				win = self.play_once_safe(agtB, agtA, 1)
				score[n] += (-1) * win
		print()
		return score

	def play_once_safe(self, agtA, agtB, who = 0):  # play with upper limit on moves
		HB = HexBoard(self.bSize)
		numTurns = 0
		while HB.checkWin() == 0 and numTurns < 100:
			# Move performed by i
			yHat = agtA.forward(HB.linearBoard())
			linMove = np.argmax(yHat)
			coo = HB.cooFromLinSpace(linMove)
			if HB.move(coo[0], coo[1]):
				return -1

			if HB.checkWin() > 0:
				return HB.checkWin() * 2

			# Move performed by j
			yHat = agtB.forward([(-1) * i for i in HB.linearBoard()])
			linMove = np.argmax(yHat)
			coo = HB.cooFromLinSpace(linMove)
			if HB.move(coo[0], coo[1]):
				return 1

			if HB.checkWin() < 0:
				return HB.checkWin() * 2

			numTurns += 1
		return HB.checkWin() * 2

	def play_once_print(self, agtA, agtB):  # play with upper limit on moves
		HB = HexBoard(self.bSize)
		numTurns = 0
		while HB.checkWin() == 0 and numTurns < 100:
			# Move performed by i
			linBoard = HB.linearBoard()
			yHat = agtA.forward(linBoard)
			linMove = np.argmax(yHat)
			coo = HB.cooFromLinSpace(linMove)
			if HB.move(coo[0], coo[1]):
				print("i lost by invalid move")
				print(coo)
				return -1
			print("Move i:")
			print(HB.board)
			print()
			if HB.checkWin() > 0:
				break

			# Move performed by j
			linBoard = [(-1) * i for i in HB.linearBoard()]  # *(-1) So the NN sees enemy as negative
			yHat = agtB.forward(linBoard)
			linMove = np.argmax(yHat)
			coo = HB.cooFromLinSpace(linMove)
			if HB.move(coo[0], coo[1]):
				print("j lost by invalid move")
				print(coo)
				return 1
			print("Move j:")
			print(HB.board)
			print()

			if HB.checkWin() < 0:
				break

			numTurns += 1
		return HB.checkWin()
