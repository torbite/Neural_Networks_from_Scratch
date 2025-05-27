from neural2 import *

new = NeuralNetwork(1, [1,4,1],'sigmoid')

X= []
Y = []
for i in range(1000):
    X.append([i])
    Y.append(0 if i<3 else 1)

print(X)
print(Y)
for i in range(100000):
    new.train( X, Y, 0.0005)
for i in range(10):
    print(new([0]))

with open('neural_network.pkl', 'wb') as f:
    pickle.dump(new, f)
