import numpy as np
import numpy.random as rand
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import chopsticks as cs
AI = cs.net([4,6,3],"Jill")

def percentage ():
    percent = 0
    for i in range(148):
        result = AI.fire([inputs[i]])
        thing = [0,0,0]
        thing[result[0].argmax()] = 1
        if list(expected[i]) == list(thing):
            percent += 1
    return percent / 148 * 100


data = pd.read_csv("/Users/adenpower/Documents/Personal/Repos/AI Tests/dataset1")

data = data.values
inputs = data[0:,:4]
expected = list(data[0:,4])
def prep(expects):
    if expects == "Iris-versicolor":
        return [1,0,0]
    elif expects == "Iris-virginica":
        return [0,1,0]
    else:
        return [0,0,1]

expected = list(map(prep,expected))
print("Starting Success: {}%".format(percentage()))



epochs = 2000
error = []
for i in range(epochs):
    error.append(AI.epoch(inputs,expected))
    if i % 20 == 0:
        print("\r {} out of {} epochs".format(i,epochs),end="")

print(" ... Finsihed!")
print("Final Success: {}%".format(percentage()))

fig = plt.figure()
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(range(epochs),error)
axes.set_xlabel("Epochs")
axes.set_ylabel("Error")
axes.set_title(AI.name)
if input("Would you like to see a graph?, Type 'y' if yes...").lower() == "y":
    plt.show()


