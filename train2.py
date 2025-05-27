from neural2 import *

new = NeuralNetwork(1, [1,4,1],'Sigmoid')
X = [[1],[2],[3],[4],[5]]
Y = [0,0,1,1,1]
for i in range(100000):
    new.train( X, Y, 0.0005)
print(new([0]))

with open('neural_network.pkl', 'wb') as f:
    pickle.dump(new, f)
