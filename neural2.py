import numpy as np
import copy, pickle, math

def sumx(lst, a, b, leng):
    if a >= leng:
        return lst[a][b]
    else:
        rsp = []
        for i in range(len(lst[a][b])):
            for x in sumx(copy.deepcopy(lst), a+1, i, leng):
                rsp.append(copy.deepcopy(lst[a][b][i])*x)
        return rsp
            
def getTill(neu, a, b, c, xi, d = 0):
    if a == 0:
        return neu.layers[a].layer[b](xi)
    else:
        if d == 0:
            return getTill(neu, a-1, c, 0, xi, d+1)
        else:
            coiso = []
            for fct_i in range(len(neu.layers[a].layer[b].w)):
                coiso.append(getTill(neu, a-1, fct_i, 0, xi, d+1))
            return neu.layers[a].layer[b](coiso)

def thingsGiver(neu, layer_num, fct_num):
    things = []
    for inv in range(len(neu.layers)-1,layer_num, -1):
        z = []
        for fct in neu.layers[inv].layer:

            if layer_num + 1 != inv:
                z.append(copy.deepcopy(fct.w))
            else:
                z.append([copy.deepcopy(fct.w[fct_num])])
        things.append(z)
    return things

def relu(z):
    if z > 0:
        return z
    else:
        return 0
    
def no_actv(z):
    return z

#def sigmoid(z):
#    return 1/1 + math.e**(-z)



def sigmoid(z):
    z = np.array(z)
    #z_max = np.max(z)
    #exp_values = np.exp(z - z_max)
    return 1 / (1 + np.exp(-z))


class Function():
    def __init__(self, w : list, b : float, activation):
        self.w = w
        self.b = b
        self.activation = activation
        
    
    def __call__(self, x : list):
        #print('in' ,x)
        if len(x) != len(self.w):
            print(f'x={len(x)}, w={len(self.w)}')
            raise ValueError("The x vector and w vector have different lengths")
        n = len(self.w)
        wx_product = 0

        for i in range(n):
            wx_product += self.w[i] * x[i]
        # print('wx_product', wx_product, self.w, x)
        result = wx_product + self.b
        result = self.activation(result)
        #print('a_out' ,result, self.activation)
        return result


class Dense_layer():

    def __init__(self, input_shape, output_shape, actvation):
        self.actvations = {'RELU': relu, 'None' : no_actv, 'sigmoid':sigmoid}
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer = [Function([1] * input_shape, 1, self.actvations[actvation])] * output_shape
        self.actvation = actvation
        
    
    def __call__(self, a_in):
        if len(a_in) != self.input_shape:
            raise ValueError('The a_in length is not equals to the input shape')
        a_out = [0] * self.output_shape
        n = len(self.layer)
        for i in range(n):
            a_out[i] = self.layer[i](a_in)
        return a_out

class NeuralNetwork():
    def __init__(self, input_shape : int, output_shapes : list, output_activation = 'None', activations = 'RELU'):
        
        layers = []
        for i in range(len(output_shapes)):
            if i == 0:
                layers.append(Dense_layer(input_shape, output_shapes[i], activations))
            elif i == len(output_shapes)-1:
                layers.append(Dense_layer(output_shapes[i-1], output_shapes[i], output_activation))
            else:
                layers.append(Dense_layer(output_shapes[i-1], output_shapes[i],  activations))
            
        self.layers = layers
        
    def __call__(self, a_in):
        n = len(self.layers)
        for i in range(n):
            layer = self.layers[i]
            a_out = layer(a_in)
            if i < n - 1:
                a_in = a_out
        return a_out
    
    def train(self, X, Y, alpha):
        for layer_i in range(len(self.layers)):
            
            for fct_i in range(len(self.layers[layer_i].layer)):
                things = thingsGiver(self, layer_i, fct_i)
                if(len(things) > 0):
                    suma = sum(sumx(copy.deepcopy(things),0,0,len(things)-1))
                else:
                    suma = 1


                sumaturio_b = 0
                for i in range(len(X)):
                    
                    sumaturio_b += (self(X[i])[0] - Y[i]) * (suma)
                b_new = self.layers[layer_i].layer[fct_i].b - alpha * sumaturio_b
                ws_new = []
                for w_i in range(len(self.layers[layer_i].layer[fct_i].w)):
                    if layer_i != 0:
                        sumaturio = 0
                        for i in range(len(X)):
                            sumaturio += (self(X[i])[0] - Y[i]) * (suma) * getTill(self, layer_i, fct_i, w_i, X[i], 0)
                        w_new = self.layers[layer_i].layer[fct_i].w[w_i] - alpha * sumaturio
                        ws_new.append(copy.deepcopy(w_new))
                    else:
                        sumaturio = 0
                        for i in range(len(X)):
                            sumaturio += (self(X[i])[0] - Y[i]) * (suma) * X[i][0]
                        w_new = self.layers[layer_i].layer[fct_i].w[w_i] - alpha * sumaturio
                        ws_new.append(copy.deepcopy(w_new))
                self.layers[layer_i].layer[fct_i].w = copy.deepcopy(ws_new)
                self.layers[layer_i].layer[fct_i].b = copy.deepcopy(b_new)

if __name__ == "__main__":
    neu = NeuralNetwork(1, [1],'None')

    for layer in neu.layers:
        for layer in layer.layer:
            print(layer.w, layer.b, layer.activation)

    X = []
    Y = []
    for i in range(100):
        X.append([i])
        Y.append(i*2)
    for i in range(1001):
        neu.train( X, Y, 0.0000001)
        print(neu([1]))
        if i % 100 == 0:
            print(f"Iteration {i}:")
            for i in range(10):
                print(i, neu([i]))
    
    for layer in neu.layers:
        for layer in layer.layer:
            print(layer.w, layer.b, layer.activation)

    with open('neural_network.pkl', 'wb') as f:
        pickle.dump(neu, f)