import math
from classification_test import class_balance_test, dataset_test

mean1 = 0
deviation1 = math.sqrt(1)
mean2 = 0
deviation2 = math.sqrt(0.25)
gauss_distr = [mean1, deviation1, mean2, deviation2]
# class_balance_test(gauss_distr)
dataset_test(gauss_distr, classifier='bayes', validation='resub', sample_estimate=False)
