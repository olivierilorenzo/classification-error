import math
import numpy as np
from openpyxl import load_workbook
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
from bayes_classifier import bayes_rule
from classifier_selector import ClassifierSelector


def class_balance_test(gauss_distr):
    wb = load_workbook("error-estimates.xlsx")
    sheet1 = wb['Foglio1']
    row = 4
    column = 2
    test_list = [[100, 100], [100, 200], [200, 100], [200, 200], [200, 400], [400, 200], [500, 500], [500, 1000], [1000, 500], [1000, 1000]]
    mu1 = gauss_distr[0]
    sigma1 = gauss_distr[1]
    mu2 = gauss_distr[2]
    sigma2 = gauss_distr[3]
    x1 = np.random.normal(mu1, sigma1, 1000)
    x2 = np.random.normal(mu2, sigma2, 1000)
    for test in test_list:
        x = np.concatenate((x1[:test[0]], x2[:test[1]]), axis=0)
        y1 = np.zeros(test[0])
        y2 = np.full(test[1], 1)
        y = np.concatenate((y1, y2), axis=0)

        p1 = test[0]/(test[0]+test[1])  # prior probability
        p2 = test[1] / (test[0] + test[1])
        y_pred, e1, e2 = bayes_rule(x, mu1, sigma1, mu2, sigma2, p1, p2)

        conf_matrix = metrics.confusion_matrix(y, y_pred)
        tmp1 = conf_matrix[0, 1] / list(y).count(0)
        tmp2 = conf_matrix[1, 0] / list(y).count(1)
        tmp = (conf_matrix[0, 1] + conf_matrix[1, 0]) / len(y)
        sheet1.cell(row=row, column=column).value = e1
        sheet1.cell(row=row + 1, column=column).value = e2
        sheet1.cell(row=row + 2, column=column).value = (e1+e2)
        sheet1.cell(row=row + 3, column=column).value = tmp1
        sheet1.cell(row=row + 4, column=column).value = tmp2
        sheet1.cell(row=row + 5, column=column).value = tmp
        wb.save("error-estimates.xlsx")
        column += 1


def dataset_test(gauss_distr, classifier, validation, sample_estimate=False):
    wb = load_workbook("error-estimates.xlsx")
    sheet1 = wb['Foglio1']
    row = 29
    column = 3
    error1 = []
    error2 = []
    error = []
    e1, e2, tmp1, tmp2, tmp = 0, 0, 0, 0, 0
    test_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    mu1 = gauss_distr[0]
    sigma1 = gauss_distr[1]
    mu2 = gauss_distr[2]
    sigma2 = gauss_distr[3]
    clf = ClassifierSelector(classifier)

    for test in test_list:
        for i in range(10):
            x1 = np.random.normal(mu1, sigma1, test)
            x2 = np.random.normal(mu2, sigma2, test)
            x = np.concatenate((x1, x2), axis=0)
            y1 = np.zeros(test)
            y2 = np.full(test, 1)
            y = np.concatenate((y1, y2), axis=0)

            if validation == 'resub':
                if sample_estimate:
                    mu1 = np.mean(x1)
                    mu2 = np.mean(x2)
                    sigma1 = math.sqrt(np.var(x1))
                    sigma2 = math.sqrt(np.var(x2))
                if classifier == 'bayes':
                    y_pred, e1, e2 = bayes_rule(x, mu1, sigma1, mu2, sigma2, 0.5, 0.5)
                else:
                    clf.fit(x, y)
                    y_pred = clf.predict(x)
                conf_matrix = metrics.confusion_matrix(y, y_pred)
                tmp1 = conf_matrix[0, 1] / list(y).count(0)
                tmp2 = conf_matrix[1, 0] / list(y).count(1)
                tmp = (conf_matrix[0, 1] + conf_matrix[1, 0]) / len(y)

            if validation == 'holdup':
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0, stratify=y)

                if sample_estimate:
                    x1 = x_train[y_train == 0]
                    x2 = x_train[y_train == 1]
                    mu1 = np.mean(x1)
                    mu2 = np.mean(x2)
                    sigma1 = math.sqrt(np.var(x1))
                    sigma2 = math.sqrt(np.var(x2))

                if classifier == 'bayes':
                    y_pred, e1, e2 = bayes_rule(x_test, mu1, sigma1, mu2, sigma2, 0.5, 0.5)
                else:
                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test)

                conf_matrix = metrics.confusion_matrix(y_test, y_pred)
                tmp1 = conf_matrix[0, 1] / list(y_test).count(0)
                tmp2 = conf_matrix[1, 0] / list(y_test).count(1)
                tmp = (conf_matrix[0, 1] + conf_matrix[1, 0]) / len(y_test)

            if validation == 'cross':
                cross1 = []
                cross2 = []
                cross = []
                x = np.reshape(x, [len(x), 1])
                skf = StratifiedKFold(n_splits=10)
                skf.get_n_splits(x, y)

                for train_index, test_index in skf.split(x, y):
                    x_train, x_test = x[train_index], x[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    if sample_estimate:
                        x1 = x_train[y_train == 0]
                        x2 = x_train[y_train == 1]
                        mu1 = np.mean(x1)
                        mu2 = np.mean(x2)
                        sigma1 = math.sqrt(np.var(x1))
                        sigma2 = math.sqrt(np.var(x2))

                    if classifier == 'bayes':
                        y_pred, e1, e2 = bayes_rule(x_test, mu1, sigma1, mu2, sigma2, 0.5, 0.5)
                    else:
                        clf.fit(x_train, y_train)
                        y_pred = clf.predict(x_test)

                    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
                    c1 = conf_matrix[0, 1] / list(y_test).count(0)
                    c2 = conf_matrix[1, 0] / list(y_test).count(1)
                    c = (conf_matrix[0, 1] + conf_matrix[1, 0]) / len(y_test)
                    cross1.append(c1)
                    cross2.append(c2)
                    cross.append(c)

                tmp1 = np.average(cross1)
                tmp2 = np.average(cross2)
                tmp = np.average(cross)
            error1.append(tmp1)
            error2.append(tmp2)
            error.append(tmp)

        sheet1.cell(row=row, column=column).value = np.average(error1)
        sheet1.cell(row=row + 1, column=column).value = np.var(error1)
        sheet1.cell(row=row + 2, column=column).value = np.average(error2)
        sheet1.cell(row=row + 3, column=column).value = np.var(error2)
        sheet1.cell(row=row + 4, column=column).value = np.average(error)
        sheet1.cell(row=row + 5, column=column).value = np.var(error)

        if test == 100:
            row1 = 14
            col1 = 2
            for er in error1:
                sheet1.cell(row=row1, column=col1).value = er
                row1 += 1
            row1 = 14
            col1 = 3
            for er in error2:
                sheet1.cell(row=row1, column=col1).value = er
                row1 += 1
            row1 = 14
            col1 = 4
            for er in error:
                sheet1.cell(row=row1, column=col1).value = er
                row1 += 1

        wb.save("error-estimates.xlsx")
        column += 1
    print("Bayes error1", e1)
    print("Bayes error2", e2)
    print("Bayes error", e1+e2)
