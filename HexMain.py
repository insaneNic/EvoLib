from backend.Hexplode import *
from backend.agent import *
from backend.funcs import softmax

game = HexGame(5)

group = TrainGroup(game, 61, 7, init_file = 'new_new9', layer_sizes = [])

group.training(51, stay_prob = 0.6,
			   shake_eps = 0.05, shake_decay = 0.99,
			   save_img = True, bounds = 0.05)

group.plot_agents_3d()

ind = np.argpartition(-group.recent_score, 2)

print('Best two ind: ' + str(ind[:2]))
print(group.recent_score[ind])

agtA = group.all_agents[ind[0]]
agtB = group.all_agents[ind[1]]

agtA.save_weights('all_new')

print("Best v Best:")
game.play_once_print(agtA, agtA)

print('\n---------')
print("Best v 2nd Best")
game.play_once_print(agtA, agtB)

HB = HexBoard(5)
yHat = agtA.forward(HB.linearBoard())
coo = HB.get_prob_board_from_lin(yHat)

np.set_printoptions(precision = 3, suppress = True)

print("starting board:")
print(coo)
