import numpy as np
import math
import scipy.stats as stats
import scipy.integrate as integrate


def bayes_border(mu1, sigma1, mu2, sigma2, prior1, prior2):
    """
    :param mu1:
    :param sigma1:
    :param mu2:
    :param sigma2:
    :param prior1: prior probability of class1
    :param prior2: prior probability of class2
    :return: bayes borders
    """
    b1, b2 = 0, 0
    # intersections between two gaussian distributions
    a = 1 / (2 * sigma1 ** 2) - 1 / (2 * sigma2 ** 2)
    b = mu2 / (sigma2 ** 2) - mu1 / (sigma1 ** 2)
    c = mu1 ** 2 / (2 * sigma1 ** 2) - mu2 ** 2 / (2 * sigma2 ** 2) - math.log((sigma2 / sigma1)*(prior1/prior2))
    if a == 0:
        if b == 0:
            # degenerate curves: a == b == c == 0, f1(x)==f2(x) for all x
            print('Curves are degenerate...')
        else:
            # single intersection: mu1 ~= mu2
            b1 = -c / b
    else:
        # two intersections; both parameters are different
        d = (b ** 2) - (4 * a * c)
        if d >= 0:
            b1 = (-b + math.sqrt(d)) / (2 * a)
            b2 = (-b - math.sqrt(d)) / (2 * a)
        else:
            print('No intersections...')

    return b1, b2


def bayes_error(mu1, sigma1, mu2, sigma2, border1, border2, prior1, prior2):
    """
    :param mu1:
    :param sigma1:
    :param mu2:
    :param sigma2:
    :param border1:
    :param border2:
    :param prior1:
    :param prior2:
    :return: calculate integrals on the bayes borders and return the bayes error on class1 and class2
    """
    result1 = integrate.quad(lambda x: stats.norm.pdf(x, mu2, sigma2), np.NINF, border1)
    result2 = integrate.quad(lambda x: stats.norm.pdf(x, mu2, sigma2), border2, np.inf)
    result3 = integrate.quad(lambda x: stats.norm.pdf(x, mu1, sigma1), border1, border2)
    error1 = result3[0]*prior1
    error2 = (result1[0] + result2[0])*prior2

    return error1, error2


def bayes_rule(data, mu1, sigma1, mu2, sigma2, prior1, prior2):
    """
    :param data: dataset X
    :param mu1:
    :param sigma1:
    :param mu2:
    :param sigma2:
    :param prior1:
    :param prior2:
    :return: prediction made on the bayes border, return also bayes borders and bayes error(e1+e2)
    """
    predict = []
    border1, border2 = bayes_border(mu1, sigma1, mu2, sigma2, prior1, prior2)
    e1, e2 = bayes_error(mu1, sigma1, mu2, sigma2, border1, border2, prior1, prior2)
    for sample in data:
        if border1 < sample < border2:
            predict.append(1)
        else:
            predict.append(0)

    return predict, e1, e2, border1, border2
