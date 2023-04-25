import numpy as np
import matplotlib.pyplot as plt


def h_softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)

def d_softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))


functions = {
    "tanh":{
        "function": lambda x: np.tanh(x),
        "derivative": lambda x: (1 - np.square(np.tanh(x)))
    },
    "relu":{
        "function": lambda x: np.where(x > 0, x, 0),
        "derivative": lambda x: np.where(x > 0, 1, 0)
    },
    "sigmoid":{
        "function": lambda x: (1/(1 + np.exp(-x))),
        "derivative": lambda x: ( 1/(1 + np.exp(-x)) )*( 1 - (1/(1 + np.exp(-x))) )
    },
    "softmax":{
        "function": lambda x : np.apply_along_axis(h_softmax, 1, x),
        "derivative": lambda x : np.apply_along_axis(d_softmax, 1, x)
    }
}


# Implementa una red neuronal con una hidden layer
class MLP():
    # sh: numero de hidden neurons.
    # s1: funcion para usar en el hidden layer.
    # s2: funcion para usar en el output layer.
    def __init__(self, inputs, outputs, sh, s1 = "tanh", s2 = "tanh"):
        self.x = inputs
        self.z = outputs

        self.p = len(inputs)
        self.si = len(inputs[0])
        self.so = len(outputs[0])
        self.sh = sh

        self.y0 = np.zeros((self.p, self.si + 1))
        self.y1 = np.zeros((self.p, self.sh + 1))
        self.y2 = np.zeros((self.p, self.so))

        self.w1 = np.random.normal(0, 0.5, (self.si + 1, sh))
        self.w2 = np.random.normal(0, 0.5, (sh + 1, self.so))

        self.s1 = functions.get(s1)["function"]
        self.ds1 = functions.get(s1)["derivative"]

        self.s2 = functions.get(s2)["function"]
        self.ds2 = functions.get(s2)["derivative"]

        self.error = 10000000

    def bias_add(self, v):
        bias = np.ones((len(v), 1))
        return np.hstack([v, bias])

    def bias_sub(self, v):
        return v[:,:-1]
 
    def forward_propagate(self):
        self.y0 = self.bias_add(self.x)
        self.y1[:] = self.bias_add(self.s1(self.y0 @ self.w1))
        self.y2[:] = self.s2(self.y1 @ self.w2)
    
    def propagate_error(self, eta):
        e2 = self.z - self.y2
        d2 = e2*self.ds2(self.y1 @ self.w2)
        delta_w2 = eta*(self.y1.T @ d2)

        e1 = d2 @ self.w2.T
        d1 = self.bias_sub(e1 * self.bias_add(self.ds1(self.y0 @ self.w1)))
        delta_w1 = eta*(self.y0.T @ d1)

        return delta_w1, delta_w2
    
    def back_propagate(self, eta):
        delta_w1, delta_w2 = self.propagate_error(eta)
        self.w1 += delta_w1
        self.w2 += delta_w2
     
    def train(self, eta = 0.0001, minError = 0.01, maxIter = 100):
        error = 10000
        epoch = 0
        errores = []

        while(error > minError and epoch < maxIter):
            self.forward_propagate()
            currError = self.z - self.y2 
            self.back_propagate(eta)
            error = np.mean(np.square(currError))
            errores.append(error)
            epoch += 1
            if (epoch % 100 == 0):
                print("Época: ", epoch)

        self.error = error
        return errores
    
    def predict(self, x):
        y0 = self.bias_add(x)
        y1 = self.bias_add(self.s1(y0 @ self.w1))
        y2 = self.s2(y1 @ self.w2)

        return y2
    
    def get_predictions(self):
        return self.y2


