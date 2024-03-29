import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


class ClassifierSelector:
    classifiers = [  # classifiers parameters settings
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
        if len(x_train.shape) == 1:  # when using the sample generated dataset(single feature)
            x_train = np.reshape(x_train, [len(x_train), 1])
        self.classifier.fit(x_train, y_train)

    def predict(self, x_test):
        if len(x_test.shape) == 1:
            x_test = np.reshape(x_test, [len(x_test), 1])
        y_test = self.classifier.predict(x_test)
        return y_test


def dataset_loader(n_samples):
    """
    :param n_samples: draws n_samples per class random
    :return: data x and labels y, result dataset is not shuffled(first class1 followed by class2)
    """
    data = pd.read_csv("bank-additional-full.csv", delimiter=";")
    mapping_dict = {"y": {"no": 0, "yes": 1}}
    data.replace(mapping_dict, inplace=True)
    y = data.values[:, 20]
    data = data.drop(columns="y")
    df = pd.get_dummies(data, columns=["job", "marital", "education", "default", "housing", "loan", "contact", "month",
                                       "day_of_week", "poutcome"])
    x = df.values
    x = np.array(x)
    y = np.array(y, dtype=int)
    x1 = x[y == 0]  # only class1 elements
    x2 = x[y == 1]  # only class2 elements
    idx1 = np.random.randint(len(x1), size=n_samples)
    idx2 = np.random.randint(len(x2), size=n_samples)
    x1 = x1[idx1, :]  # random draws class1 elements from total class1
    x2 = x2[idx2, :]
    x = np.concatenate((x1, x2), axis=0)
    y1 = np.zeros(n_samples)
    y2 = np.full(n_samples, 1)
    y = np.concatenate((y1, y2), axis=0)
    return x, y


def plot_hist(x1, x2, info):  # plot histogram of the sample distribution on class1 and class2
    """
    data = [x1, x2]

    colors = ["red", "blue"]
    classes = ["class1", "class2"]
    fig, ax = plt.subplots()

    for color, x, label in zip(colors, data, classes):
        ax.scatter(x, y, c=color, s=10, label=label, alpha=0.3, edgecolors='none')
    ax.legend()

    plt.axvline(-0.6797779934458726, color="red")
    plt.axvline(0.6797779934458726, color="red")
    """
    n_bins = 100
    plt.hist(x1, bins=n_bins)

    plt.hist(x2, bins=n_bins)

    # plt.xticks(np.arange(min(x), max(x)+1, 1))
    plt.savefig('plots/samples-{}-{}-{}.png'.format(info[0], info[1], info[2]))
    plt.clf()


def plot_error(error_list, bayes, info):  # plot the scatter error of the ten tests, along with mean error and bayes err
    x = error_list
    y = np.zeros(len(x))
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='green', s=100, alpha=0.3, edgecolors='none')
    plt.axvline(np.mean(x), color="blue")
    # plt.axvline(bayes, color="red")
    plt.text(np.mean(x), -0.010, np.mean(x))
    # plt.text(bayes, -0.005, bayes)
    plt.savefig('plots/error-{}-{}-{}.png'.format(info[0], info[1], info[2]))
    plt.clf()


def plot_distr(mu1, sigma1, mu2, sigma2, b1, b2, info):  # plot two distribution based on parameters
    t = np.linspace(mu1 - 3 * sigma1, mu1 + 3 * sigma1, 100)
    plt.plot(t, 0.5 * stats.norm.pdf(t, mu1, sigma1), color='green')
    plt.plot(t, 0.5 * stats.norm.pdf(t, mu2, sigma2), color='red')
    plt.axvline(b1)
    plt.axvline(b2)
    plt.text(b1, 0, b1)
    plt.text(b2, 0.10, b2)
    plt.savefig('plots/distr-{}-{}-{}.png'.format(info[0], info[1], info[2]))
    plt.clf()
