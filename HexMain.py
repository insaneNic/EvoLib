from backend.Hexplode import *
from backend.agent import *
from backend.funcs import softmax

game = HexGame(5)

group = TrainGroup(game, 81, 7, [game.hexNum, 80, 80, 80, game.hexNum], softmax, 3.)

group.training(96, stay_prob = 0.5,
			   shake_eps = 0.2, shake_decay = 0.96,
			   save_img = True, bounds = 0.05)

group.plot_agents_3d()

ind = np.argpartition(-group.recent_score, 2)

print('Best two ind: ' + str(ind[:2]))
print(group.recent_score[ind])

agtA = group.all_agents[ind[0]]
agtB = group.all_agents[ind[1]]

game.play_once_print(agtA, agtB)

HB = HexBoard(5)
yHat = agtA.forward(HB.linearBoard())
coo = HB.get_prob_board_from_lin(yHat)

np.set_printoptions(precision = 3, suppress = True)

print("starting board:")
print(coo)

agtA.save_weights('test_agent')