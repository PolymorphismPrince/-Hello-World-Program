import numpy as np
import numpy.random as rand
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
alphabet = "abcdefghijklmnopqrstuvwxyz"

class net:
    learning_rate = 0.003
    """
    The net class describes a feed forward neural network

    A net uses a backpropagation algorithim, it can have any number of layers and any number of neurons per layer
    """
    def __init__ (self,shape,name="Bob"):
        self.input = shape[0]
        self.shape = shape[1:]
        self.weights = []
        self.biasses = []
        self.name = name
        self.seedName(name)
        for l in range(len(self.shape)):
            self.weights.append(rand.rand(self.shape[l],shape[l]) * 2 - 1)
            self.biasses.append(rand.rand(self.shape[l]) * 2 - 1)

    def seedName (self, letters):
        l = len(letters)
        output = 0
        for i in range(len(letters)):
            output += (26 ** (l - i - 1)) * alphabet.index(letters[i].lower())
        rand.seed(output) 
    
    def fire (self,inputs,backprop = False):
        results = [inputs]
        for l in range(len(self.shape)):
            inputs = np.dot(inputs,np.transpose(self.weights[l])) + self.biasses[l]
            results.append(inputs.copy())
            inputs = self.activate(inputs)
        if backprop:
            return np.array(results)
        else:
            return inputs
    
    def epoch (self,inputs,outputs):
        results = []
        for input in inputs:
            results.append(self.fire(np.array(input),True))
        return self.backprop(results,outputs)
    
    def learn(self,inputs,outputs,epochs = 1,batch_size = "all"):
        in_batches = np.array([])
        out_batches = np.array([])
        if batch_size != "all":
            for i in range(0,len(inputs),batch_size):
                if i + batch_size <= len(inputs) - 1:
                    in_batches = np.append(in_batches,np.array(inputs[i:i+batch_size - 1]))
                    out_batches = np.append(out_batches,np.array(outputs[i:i+batch_size - 1]))
                else:
                    in_batches = np.append(in_batches,np.array(inputs[i:]))
                    out_batches = np.append(out_batches,np.array(outputs[i:]))
        else:
            np.append(in_batches,np.array(inputs))
            np.append(out_batches,np.array(outputs))
        error = []
        for i in range(epochs):
            error.append(self.epoch(rand.choice(in_batches),rand.choice(out_batches)))
        return error

    def backprop (self, ins, outs):
        new_weights = self.weights.copy()
        new_weights = [i * 0 for i in new_weights]
        new_biasses = self.biasses.copy()
        new_biasses = [i * 0 for i in new_biasses]
        for i in range(len(ins)):
            inputs = ins[i]
            outputs = np.array(outs[i])
            if i == 0:
                 error = np.sum((self.activate(inputs[-1]) - outputs) ** 2)
            error = (error + np.sum((self.activate(inputs[-1]) - outputs) ** 2)) / 2
            derivs = np.array(2 * self.activate(inputs[-1]) - 2 * outputs) 
            for l in range(len(self.biasses) - 1,-1,-1):
                derivs = derivs * self.derive(inputs[l + 1])
                new_biasses[l] = new_biasses[l] + derivs * -1 * self.learning_rate
                for n in range(len(self.weights[l])):
                    new_weights[l][n] = new_weights[l][n] + self.activate(inputs[l]) * derivs[n] * self.learning_rate * -1
                derivsTemp = derivs.copy()
                if l > 0:
                    derivs = np.zeros(len(self.weights[l-1]))
                    for n in range(len(self.weights[l])):   
                        derivs += self.weights[l][n] * derivsTemp[n]
        
        for l in range(len(self.weights)):
            self.weights[l] = self.weights[l] + new_weights[l] / len(ins)
            self.biasses[l] = self.biasses[l] +  new_biasses[l] / len(ins)
        return error

    def activate (self,x):
        return x

    def derive (self,x):
        return x/x

"""
AI = net([4,6,3],"Jill")
data = pd.read_csv("/Users/adenpower/Documents/Personal/Repos/AI Tests/dataset1")

data = data.values
ins = data[0:,:4]
expected = list(data[0:,4])
def prep(expects):
    if expects == "Iris-versicolor":
        return [1,0,0]
    elif expects == "Iris-virginica":
        return [0,1,0]
    else:
        return [0,0,1]

expected = list(map(prep,expected))

AI.learn(ins,expected,100)

"""