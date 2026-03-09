import matplotlib.pyplot as plt
import numpy as np
import random
import json

with open("./train.json" , mode = 'r' ) as file:
    train = json.load(file)
    
data = np.array(train)
print(data.shape)