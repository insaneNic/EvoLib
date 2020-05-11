from backend.StockGame import *
from backend.agent import *
from backend.funcs import *

game = StockGame(mu = 0., sigma = 0.1)

group = TrainGroup(game, 71, 6, [game.in_size, 12, 12, 12, game.out_size], linear, 4.0)

group.training(200, stay_prob = 0.8, shake_eps = 0.15, save_img = True, bounds = 0.5)

group.plot_agents_3d()

ind = np.argpartition(-group.recent_score, 2)

print('Best two ind: ' + str(ind[:2]))
print(group.recent_score[ind])

for _ in range(3):
	results, money, price, holdings, hidden = game.play_print(group.all_agents)

	fig, ax1 = plt.subplots()
	color = 'tab:red'
	ax1.set_xlabel('time')
	ax1.set_ylabel('Stock', color = color)
	ax1.plot(price, color = color)
	ax1.tick_params(axis = 'y', labelcolor = color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('Trades', color = color)  # we already handled the x-label with ax1
	ax2.plot(holdings[:, ind[0]], color = color)
	ax2.plot(money[:, ind[0]], color = 'tab:green')
	ax2.tick_params(axis = 'y', labelcolor = color)

	fig.tight_layout()
	plt.show()

	# plt.plot(hidden[:, ind[0]])
	# plt.show()
