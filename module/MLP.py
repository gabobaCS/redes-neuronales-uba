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
    },
    "x":{
        "function": lambda x : np.sin(x),
        "derivative": lambda x : np.cos(x)
    }
}


# Implementa una red neuronal con múltiples hidden layers
class MLP():
    # sh: numero de hidden neurons.
    # lh: numero de hidden neurons por layer.
    # s1: funcion para usar en el hidden layer.
    # s2: funcion para usar en el output layer.
    # sh: lista de funciones de activacion
    def __init__(self, inputs, outputs, lh, fh = None, s1 = "tanh", s2 = "tanh"):
        # np.random.seed(1) #solo para dev
        self.inputs = inputs
        self.outputs = outputs
        self.x = inputs
        self.z = outputs

        self.p = len(inputs)
        self.si = len(inputs[0])
        self.so = len(outputs[0])
        self.lh = lh
        if fh == None:
            fh = ["tanh" for i in range(len(lh))] # funcion de activacion para las hidden layers
            fh.append("tanh") # funcion de activacion para el output layer
        self.fh = fh
        
        # Inicializamos los hidden layers
        y = [np.zeros((self.p, self.si + 1))]
        for i in range(len(lh)):
            y.append(np.zeros((self.p, self.lh[i] + 1)))
        y.append(np.zeros((self.p, self.so)))
        self.y = y
        
        # Inicializamos las matrices de pesos
        w = [None, np.random.normal(0, 0.5, (self.si + 1, self.lh[0]))]
        for i in range(len(lh) - 1):
            w.append(np.random.normal(0, 0.5, (self.lh[i] + 1, self.lh[i + 1])))
        w.append(np.random.normal(0, 0.5, (self.lh[-1] + 1, self.so)) ) 
        self.w = w

        # self.s1 = functions.get(s1)["function"]
        # self.ds1 = functions.get(s1)["derivative"]

        # self.s2 = functions.get(s2)["function"]
        # self.ds2 = functions.get(s2)["derivative"]

        self.error = 10000000

    def bias_add(self, v):
        bias = np.ones((len(v), 1))
        return np.hstack([v, bias])

    def bias_sub(self, v):
        return v[:,:-1]
 
    def forward_propagate(self): # activation
        self.y[0] = self.bias_add(self.x)
        for i in range(1, len(self.y) - 1):
            fi = functions.get(self.fh[i-1])["function"]
            self.y[i] = self.bias_add(fi(self.y[i-1] @ self.w[i]))
        fo = functions.get(self.fh[-1])["function"]
        self.y[-1] = fo(self.y[-2] @ self.w[-1])
    
    def propagate_error(self, eta): #correction
        e = self.z - self.y[-1]
        dfo = functions.get(self.fh[-1])["derivative"]
        dy = dfo(self.y[-1])
        d = e * dy
        delta_w = []
        
        for i in range(len(self.y) - 1, 0, -1):
            delta_w.insert(0, eta*(self.y[i-1].T @ d))
            e = d @ self.w[i].T
            dfi = functions.get(self.fh[i-1])["derivative"]
            dy = dfi(self.y[i-1])
            d = self.bias_sub(e * dy)

        return delta_w
    
    def back_propagate(self, eta): # adaptation
        delta_w = self.propagate_error(eta)
        for i in range(len(delta_w)):
            self.w[i+1] += delta_w[i]
            

    def train(self, eta = 0.0001, minError = 0.01, maxIter = 100, batchSize = None):
        if batchSize == None:
            batchSize = self.p 
        error = 10000
        epoch = 0
        errores = []
        
        while(error > minError and epoch < maxIter):
            order = np.random.permutation(self.p)
            error = 0
            for h in range(0,self.p,batchSize):
                increment = min(h+batchSize, self.p - 1)
                batch = order[h:increment]
                self.x = self.inputs[batch]
                self.z = self.outputs[batch]

                self.forward_propagate()
                currError = self.z - self.y[-1]
                self.back_propagate(eta)
                error += np.sum(np.square(currError))
                
            epoch += 1                
            if (epoch % 100 == 0):
                print("Época: ", epoch)
            error /= self.p
            errores.append(error)
            self.error = error
        return errores
    
    def predict(self, x):
        y = self.bias_add(x)
        for i in range(len(self.lh)):
            fi = functions.get(self.fh[i])["function"]
            y = self.bias_add(fi(y @ self.w[i+1]))
        fo = functions.get(self.fh[-1])["function"]
        y = fo(y @ self.w[-1])

        return y
    
    def get_predictions(self):
        return self.y