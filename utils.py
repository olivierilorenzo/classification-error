import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


class ClassifierSelector:
    classifiers = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(max_depth=5),
        MLPClassifier(alpha=1, max_iter=1000)
        ]
    classifier = 0

    def __init__(self, chosen_clf):
        if chosen_clf == 'kNN':
            self.classifier = self.classifiers[0]
        if chosen_clf == 'Tree':
            self.classifier = self.classifiers[1]
        if chosen_clf == 'MLP':
            self.classifier = self.classifiers[2]

    def fit(self, x_train, y_train):
        x_train = np.reshape(x_train, [len(x_train), 1])
        self.classifier.fit(x_train, y_train)

    def predict(self, x_test):
        x_test = np.reshape(x_test, [len(x_test), 1])
        y_test = self.classifier.predict(x_test)
        return y_test


def dataset_loader():
    data = pd.read_csv("bank-additional-full.csv", delimiter=";")
    mapping_dict = {"y": {"no": 0, "yes": 1}}
    data.replace(mapping_dict, inplace=True)
    y = data.values[:, 20]
    data = data.drop(columns="y")
    df = pd.get_dummies(data, columns=["job", "marital", "education", "default", "housing", "loan", "contact", "month",
                                       "day_of_week", "poutcome"])
    x = df.values

    return x, y
