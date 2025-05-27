import pickle
from neural2 import *

with open('neural_network.pkl', 'rb') as f:
    loaded_new = pickle.load(f)

for i in range(-100,100):
    print(i, loaded_new([i]))
