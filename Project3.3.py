# The goal of this project was to have the perceptron determine if a given patient could have heart disease based off a data sheet from kaggle
# The main factors the perceptron looked at was smoker, high BP, history of stroke, alcahol consumption, physical activity, diet, sex, and age




import numpy as np
import pandas as pd
from project3_1 import *

df = pd.read_csv('heart_disease.csv')
df = df.drop(columns = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income'], axis = 1)
df.loc[df["Diabetes"] == 2, "Diabetes"] = 1

train_data = np.array(df.iloc[0:100, 1:15].values.tolist())
train_answer = np.array(df.iloc[0:100, 0:1].values.tolist())
test_data = np.array(df.iloc[100:200, 1:15].values.tolist())
test_answer = np.array(df.iloc[100:200, 0:1].values.tolist())

test = percep(train_data.shape[1], learning_rate=0.0025)
print(test.weight)
test.fit(train_data, train_answer, epochs=50000)
print(test.weight)

wronga = 0
print("Training data final answers")
for (x, target) in zip(train_data, train_answer):
    answer = test.prediction(x)
    if answer != target[0]:
        wronga += 1
    print("data={}, answer={}, pred={}".format(x, target[0], answer))

wrong = 0
print("\nTest data answers")
for (x, target) in zip(test_data, test_answer):
    answer = test.prediction(x)
    if answer != target[0]:
        wrong += 1
    print("data={}, answer={}, pred={}".format(x, target[0], answer))

print('test percent correct: ', 100 - wronga)
print('final percent correct: ', 100 - wrong)
