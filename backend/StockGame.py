import numpy as np


class StockGame:
	def __init__(self, mu = 0.0, sigma = 0.2):
		self.mu = mu
		self.sigma = sigma

	def play_print(self, players):
		time_steps = np.random.binomial(10, p = 0.2) + 50

		money = np.ones((time_steps, len(players)))
		holdings = np.ones((time_steps, len(players)))
		price = np.ones(time_steps)
		hidden = np.zeros((time_steps, len(players)))

		change = np.zeros(len(players))

		for t in range(1, time_steps):
			for k, agt in enumerate(players):
				temp = agt.forward([money[t - 1, k], holdings[t - 1, k], price[t - 1], hidden[t - 1, k]])
				change[k] = temp[0]
				hidden[t, k] = hidden[t - 1, k] + temp[1]

			change = np.minimum(np.maximum(change, -holdings[t - 1]), (money[t-1] + 1.) / price[t - 1]) / 2
			money[t] = money[t - 1] - change * price[t - 1]
			holdings[t] = holdings[t - 1] + change

			price[t] = price[t - 1] * np.exp(np.random.normal(self.mu, 0.2) - 0.5 * self.sigma ** 2) * \
					   np.power(1.02, np.sum(change))

		return money[time_steps - 1] + holdings[time_steps - 1] * price[time_steps - 1], \
			   money, price, holdings, hidden

	def play_once(self, players):
		time_steps = np.random.binomial(10, p = 0.2) + 50

		# Initialize inputs to agent
		money = np.ones(len(players))
		holdings = np.zeros(len(players))
		price = np.ones(time_steps)
		hidden = np.zeros(len(players))

		# Change in holdings every time step
		change = np.zeros(len(players))

		# Go through time
		for t in range(1, time_steps):
			for k, agt in enumerate(players):
				temp = agt.forward([money[k], holdings[k], price[t - 1], hidden[k]])
				change[k] = temp[0]
				hidden[k] = hidden[k] + temp[1]

			# Limit the buying and selling to limit negative money
			change = np.minimum(np.maximum(change, -holdings), (money + 1.) / price[t - 1]) / 2
			money = money - change * price[t - 1]
			holdings = holdings + change

			# update price
			price[t] = price[t - 1] * np.exp(np.random.normal(self.mu, 0.2) - 0.5 * self.sigma ** 2) * \
					   np.power(1.02, np.sum(change))

		return money #+ holdings * price[time_steps - 1]

	def play(self, players):
		score = np.zeros(len(players))
		for _ in range(10):
			score += self.play_once(players)
		return score / 10.
