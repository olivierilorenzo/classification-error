import math
from classification_test import class_balance_test, dataset_test


"""
Il class_balance_test effetua una serie di test aumentando il numero di sample, ma inserendo degli esperimenti
dove i sample per classe sono diversi tra di loro, quindi con diverse prior probability.
E' un test aggiuntivo che non era richiesto nel lavoro assegnato.
Il dataset_test corrisponde ai test descritti nella relazione; consultabile nel file classification_test.py
"""

mean1 = 0
deviation1 = math.sqrt(1)
mean2 = 0
deviation2 = math.sqrt(0.25)
gauss_distr = [mean1, deviation1, mean2, deviation2]
# class_balance_test(gauss_distr)
dataset_test(classifier='bayes', validation='resub', sample_estimate=False, shuffle=True, real_dataset=False)
