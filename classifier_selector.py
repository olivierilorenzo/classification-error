import numpy as np
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
