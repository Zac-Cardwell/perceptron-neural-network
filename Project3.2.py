import numpy as np
from AI_project3_1 import *

# 0 = Dark pixel and 1 = light pixel
train_data = np.array([[0,0,0,0], [1,0,1,0], [1,1,0,1],[1,1,0,0], [1,1,1,0], [0,1,0,1], [0,0,1,0], [1,0,0,1], [1,0,0,0], [0,0,1,1]])
train_answer = np.array([[0], [1], [1], [1], [1], [1], [0], [1], [0], [1]])
test_data = np.array([[1,1,1,1], [0,1,1,1], [0,0,0,1], [0,1,0,0], [0,1,1,0], [1,0,1,1]])
test_answer = np.array([[1], [1], [0], [0], [1], [1]])


test = percep(train_data.shape[1], learning_rate=0.3)
print(test.weight)
test.fit(train_data, train_answer, epochs=200)
print(test.weight)

print("Training data final answers")
for (x, target) in zip(train_data, train_answer):
    answer = test.prediction(x)
    print("data={}, answer={}, pred={}".format(x, target[0], answer))


print("\nTest data answers")
for (x, target) in zip(test_data, test_answer):
    answer = test.prediction(x)
    print("data={}, answer={}, pred={}".format(x, target[0], answer))
