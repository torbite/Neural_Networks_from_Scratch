from neural import *
import copy, pickle


ys = [[1, 2], [2, 3]]

def sumx(lst, a, b, leng):
    if a >= leng:
        return lst[a][b]
    else:
        rsp = []
        for i in range(len(lst[a][b])):
            for x in sumx(copy.deepcopy(lst), a+1, i, leng):
                rsp.append(copy.deepcopy(lst[a][b][i])*x)
        return rsp
            
def getTill(new, a, b, c, xi, d = 0):
    if a == 0:
        return new.layers[a].layer[b](xi)
    else:
        if d == 0:
            return getTill(new, a-1, c, 0, xi, d+1)
        else:
            coiso = []
            for fct_i in range(len(new.layers[a].layer[b].w)):
                coiso.append(getTill(new, a-1, fct_i, 0, xi, d+1))
            return new.layers[a].layer[b](coiso)


def thingsGiver(new, layer_num, fct_num):
    things = []
    for inv in range(len(new.layers)-1,layer_num, -1):
        z = []
        for fct in new.layers[inv].layer:

            if layer_num + 1 != inv:
                z.append(copy.deepcopy(fct.w))
            else:
                z.append([copy.deepcopy(fct.w[fct_num])])
        things.append(z)
    return things


def train(new : NeuralNetwork, X, Y, alpha):
    for layer_i in range(len(new.layers)):
        
        for fct_i in range(len(new.layers[layer_i].layer)):
            things = thingsGiver(new, layer_i, fct_i)
            if(len(things) > 0):
                suma = sum(sumx(copy.deepcopy(things),0,0,len(things)-1))
            else:
                suma = 1


            sumaturio_b = 0
            for i in range(len(X)):
                
                sumaturio_b += (new(X[i])[0] - Y[i]) * (suma)
            b_new = new.layers[layer_i].layer[fct_i].b - alpha * sumaturio_b
            ws_new = []
            for w_i in range(len(new.layers[layer_i].layer[fct_i].w)):
                if layer_i != 0:
                    sumaturio = 0
                    for i in range(len(X)):
                        sumaturio += (new(X[i])[0] - Y[i]) * (suma) * getTill(new, layer_i, fct_i, w_i, X[i], 0)
                    w_new = new.layers[layer_i].layer[fct_i].w[w_i] - alpha * sumaturio
                    ws_new.append(w_new)
                else:
                    sumaturio = 0
                    for i in range(len(X)):
                        sumaturio += (new(X[i])[0] - Y[i]) * (suma) * X[i][0]
                    w_new = new.layers[layer_i].layer[fct_i].w[w_i] - alpha * sumaturio
                    ws_new.append(w_new)
            new.layers[layer_i].layer[fct_i].w = ws_new
            new.layers[layer_i].layer[fct_i].b = b_new
new = NeuralNetwork(1, [1,4,1])
X = [[1],[2],[3],[4]]
Y = [30,32,34,37]
for i in range(100000):
    train(new, X, Y, 0.00001)

print(new([8]))
                    
        
        
    
            