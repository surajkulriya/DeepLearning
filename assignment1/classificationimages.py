import pandas as pd
import matplotlib.pyplot as plt
from classificationClasses import oneHiddenLayer
from classificationClasses import twoHiddenLayers
from classificationClasses import getY

#Enter Path
files_path = "D:/Academics/Semester 4/CS 671 Deep Learning and Applications/Labs/Lab1/Group23/Private/data"

train_path = files_path + "/" + "image_train.csv"
valid_path = files_path + "/" + "image_valid.csv"
test_path = files_path + "/" + "image_test.csv"

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)
test_df = pd.read_csv(test_path)

train_labels = train_df["label"].to_list()
valid_labels = valid_df["label"].to_list()
test_labels = test_df["label"].to_list()

labels_dict = {"bayou":1,"desert_vegetation":2,"music_store":3}

y_train = [labels_dict[i] for i in train_labels]
y_valid = [labels_dict[i] for i in valid_labels]
y_test = [labels_dict[i] for i in test_labels]

y_train_1hot = getY(y_train)

train_df.drop(columns=["label","Unnamed: 0"], inplace=True)
valid_df.drop(columns=["label","Unnamed: 0"], inplace=True)
test_df.drop(columns=["label","Unnamed: 0"], inplace=True)

x_train = train_df.values.tolist()
x_valid = valid_df.values.tolist()
x_test = test_df.values.tolist()

model = twoHiddenLayers(32,3,0.5,12,18)
model.train(x_train, y_train_1hot, y_train, 0.3, 0.3, x_test, y_test, x_valid, y_valid, 1000, "TwohiddenLayers")