import numpy as np
class Function():
    def __init__(self, w : list, b : float):
        self.w = w
        self.b = b
    
    def __call__(self, x : list):
        if len(x) != len(self.w):
            print(f'x={len(x)}, w={len(self.w)}')
            raise ValueError("The x vector and w vector have different lengths")
        n = len(self.w)
        wx_product = 0

        for i in range(n):
            wx_product += self.w[i] * x[i]

        result = wx_product + self.b
        return result

class Dense_layer():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer = [Function([1] * input_shape, 1)] * output_shape
    
    def __call__(self, a_in):
        if len(a_in) != self.input_shape:
            raise ValueError('The a_in length is not equals to the input shape')
        a_out = [0] * self.output_shape
        n = len(self.layer)
        for i in range(n):
            a_out[i] = self.layer[i](a_in)
        return a_out

class NeuralNetwork():
    def __init__(self, input_shape : int, output_shapes : list):
        layers = []
        for i in range(len(output_shapes)):
            if i == 0:
                layers.append(Dense_layer(input_shape, output_shapes[i]))
            else:
                layers.append(Dense_layer(output_shapes[i-1], output_shapes[i]))
        self.layers = layers
        
    def __call__(self, a_in):
        n = len(self.layers)
        for i in range(n):
            layer = self.layers[i]
            a_out = layer(a_in)
            if i < n - 1:
                a_in = a_out
        return a_out
    
    
                    


    


        
