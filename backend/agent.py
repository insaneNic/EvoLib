from backend.funcs import *
import os
from json_tricks import dump, load
import numpy as np
import matplotlib.pyplot as plt
import pdb


# Puts all weights into one long vector
def agent_dist(agA, agB):
	return np.linalg.norm(np.concatenate(list(map(np.ravel, agA.weights)))
						  - np.concatenate(list(map(np.ravel, agB.weights))))


class Agent:
	def __init__(self, sizes, final_active = lambda x: x, init_spread = 1.0):
		self.nbhd_agents = []

		# Define HyperParameters
		self.layerSizes = sizes
		self.final_active = final_active

		# Weights
		self.bias = []
		self.weights = []
		for i in range(len(self.layerSizes) - 1):
			self.weights = self.weights + [
				init_spread * (np.random.rand(self.layerSizes[i + 1], self.layerSizes[i]) - 0.5)]
			self.bias = self.bias + [init_spread * (np.random.rand(self.layerSizes[i + 1], 1) - 0.5)]

	# Propagation
	def forward(self, X):
		# Propagate inputs through network
		ztemp = np.array([np.dot(self.weights[0], X)]).T
		atemp = logistic(ztemp + self.bias[0])

		for i in range(1, len(self.weights) - 1):
			ztemp = np.dot(self.weights[i], atemp)
			atemp = logistic(ztemp + self.bias[i])

		ztemp = np.dot(self.weights[len(self.weights) - 1], atemp)
		atemp = self.final_active(ztemp + self.bias[len(self.weights) - 1])

		return atemp

	def become_more_like(self, other_agent, stay_prob = 0.7):
		for k, weights in enumerate(self.weights):
			self.weights[k] = matrix_mix(weights, other_agent.weights[k], stay_prob)
			self.bias[k] = matrix_mix(self.bias[k], other_agent.bias[k], stay_prob)

	def shake(self, epsilon = 0.1):
		for k, weights in enumerate(self.weights):
			self.weights[k] = self.weights[k] + epsilon * \
							  np.random.normal(size = self.weights[k].shape) * \
							  np.random.binomial(n = 1, p = 0.2, size = self.weights[k].shape)
			self.bias[k] = self.bias[k] + epsilon * \
						   np.random.normal(size = self.bias[k].shape) * \
						   np.random.binomial(n = 1, p = 0.2, size = self.bias[k].shape)

	def get_norm(self):
		return np.linalg.norm(np.concatenate(list(map(np.ravel, self.weights))))

	def get_proj(self, dim = 2):
		full_wgt = np.concatenate(list(map(np.ravel, self.weights)))
		return np.array(list(map(np.mean, np.array_split(full_wgt, dim))))

	def save_weights(self, filename = 'agent'):
		if not os.path.exists('saves'):
			os.mkdir('saves')

		data = {'layer_sizes': self.layerSizes, 'weights': self.weights, 'bias': self.bias}
		dump(data, 'saves/' + filename + '.json')

	def load_weights(self, filename = 'agent'):
		data = load('saves/' + filename + '.json', preserve_order = False)

		self.layerSizes = data['layer_sizes']
		self.weights    = data['weights']
		self.bias       = data['bias']


class TrainGroup:
	def __init__(self, game, num_agents, nbhd_size, *args):
		# First we generate the agents, should be at least 100
		self.all_agents = [Agent(*args) for _ in range(num_agents)]
		self.num_agents = num_agents
		self.recent_score = np.zeros(num_agents)
		self.nbhd_size = nbhd_size

		self.game = game

		# Define the closest neighbors to evey agent
		self.update_nbhds()

	def update_nbhds(self):
		# Initialize matrix with infinity
		dist_matrix = np.ones((self.num_agents, self.num_agents)) * np.inf

		# Calculate distance (on upper triangle)
		for i in range(self.num_agents):
			for j in range(i, self.num_agents):
				dist_matrix[i, j] = agent_dist(self.all_agents[i], self.all_agents[j])

		# Mirror matrix
		dist_matrix = np.minimum(dist_matrix, dist_matrix.T)

		# Update entries
		for k, agt in enumerate(self.all_agents):
			agt.nbhd_agents = [(self.all_agents[i], i) for i in
							   np.argpartition(dist_matrix[k], self.nbhd_size)[:self.nbhd_size]]

	def train_step(self, stay_prob = 0.8, shake_eps = 0.01):
		# Calculate scores
		self.recent_score = 0.90 * self.game.play(self.all_agents) + 0.10 * self.recent_score

		all_indices = np.argsort(-self.recent_score).astype(int)
		# Agents Move to best agent in nbhd
		# We want to go in order of best to worst to maximize efficiency
		for k, agt in zip(all_indices, np.array(self.all_agents)[all_indices]):
			# get the index for best agent in nbhd
			indices = [i for (x, i) in agt.nbhd_agents]
			best_agent_ind = indices[np.argmax(self.recent_score[indices])]
			# become more like best agent in nbhd
			agt.become_more_like(self.all_agents[best_agent_ind], stay_prob)

		# Randomize weights for all except local best
		for k, agt in enumerate(self.all_agents):
			indices = [i for (x, i) in agt.nbhd_agents]
			best_agent_ind = indices[np.argmax(self.recent_score[indices])]
			if best_agent_ind != k:
				agt.shake(shake_eps)

	def training(self, num_steps, stay_prob = 0.8,
				 shake_eps = 0.01, shake_decay = 0.99,
				 save_img = False, bounds = 0.02):
		for k in range(num_steps):
			if save_img:
				self.plot_agents(show = False, bounds = bounds, save_name = "./imgs/epoch{0:2}".format(k))
			print("Started " + str(k))
			if k % 5 == 0:
				self.update_nbhds()
				shake_eps = shake_eps * shake_decay
			self.train_step(stay_prob, shake_eps)

	def plot_agents(self, show = True, bounds = 0.02, save_name = ''):
		plt.figure(figsize = (10, 8))
		plt.xlim(-bounds, bounds)
		plt.ylim(-bounds, bounds)
		points = np.stack([agt.get_proj() for agt in self.all_agents])
		scat = plt.scatter(points[:, 0], points[:, 1], c = self.recent_score)
		plt.legend(*scat.legend_elements(), loc = 'upper left')
		if show:
			plt.show()
		else:
			plt.savefig(save_name)
			plt.close()


if __name__ == '__main__':
	agt = Agent([4, 8, 2])
	agt.save_weights('test_agent')

	agt = Agent([])
	agt.load_weights('test_agent')
