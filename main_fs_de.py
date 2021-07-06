from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from genetic_selection import GeneticSelectionCV
import numpy as np
import pandas as pd
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time
import csv
import os

start_time = time.time()
dataset = './datasets/winsconsin_699_10_normalizado.csv'


def get_kernel(kernel_id):
    kernel_id = int(round(kernel_id))

    if kernel_id == 1:
        return 'linear'
    elif kernel_id == 2:
        return 'poly'
    elif kernel_id == 3:
        return 'rbf'
    elif kernel_id == 4:
        return 'sigmoid'
    else:
        raise Exception

def read_and_split_dataset(test_size=0.3):
    # Read the csv data into a pandas data frame (df)
    df = pd.read_csv(
        filepath_or_buffer=dataset,
        header=0
    )

    samples = df.iloc[:, :-1]  # All columns except the last
    labels = df.iloc[:, -1:]  # The last column

    # Replace all missing data (non numeric) with NaN (Not a Number)
    samples = samples.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # Create a inputer to fill missing values
    imputer = SimpleImputer(strategy='most_frequent')

    # Fill mising values in the samples
    # We only take the first two features.
    samples = imputer.fit_transform(samples)

    # Quickly sample a training set while holding out 30% of the data for testing (evaluating) our classifier
    return train_test_split(
        samples,
        labels,
        test_size=test_size,
    )

def feature_selection(samples_train, labels_train, max_features=30):
    # selector = SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False), max_features=max_features)
    selector = SelectFromModel(ExtraTreesClassifier(n_estimators=50), max_features=max_features)

    selector.fit_transform(samples_train, labels_train.values.ravel())

    return selector._get_support_mask()

def extract_features(selected_features, samples):
    indexes = []

    for i in range(len(selected_features)):
        if selected_features[i] == False:
            indexes.append(i)

    return np.delete(samples, indexes, axis=1)

def svm(X, printer=False):
    # Replace all missing data (non numeric) with NaN (Not a Number)
    samples_train_df = pd.DataFrame.from_records(samples_train).apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # Create a inputer to fill missing values
    imputer = SimpleImputer(strategy='most_frequent')

    # Fill mising values in the samples
    samples_fit = imputer.fit_transform(samples_train_df)

    svc = SVC(
        C=X[0],
        kernel=get_kernel(X[1]),
        degree=X[2],
        gamma=X[3],
        coef0=X[4],
        shrinking=X[5],
        break_ties=X[6],
    )

    # Creating a pipeline with the scaler and the classifier
    clf = make_pipeline(
        StandardScaler(),
        svc,
    )

    # Fit the classifier with the training data
    clf.fit(
        samples_fit,
        labels_train.values.ravel()  # Transform a column vector to 1d array
    )

    accuracy = clf.score(samples_test, labels_test)

    if (printer):
        # Read the csv data into a pandas data frame (df)
        df = pd.read_csv(
            filepath_or_buffer=dataset,
            header=0
        )

        samples = df.iloc[:, :-1]  # All columns except the last
        labels = df.iloc[:, -1:]  # The last column

        # Replace all missing data (non numeric) with NaN (Not a Number)
        samples = samples.apply(lambda x: pd.to_numeric(x, errors='coerce'))

        # Create a inputer to fill missing values
        imputer = SimpleImputer(strategy='most_frequent')

        # Fill mising values in the samples
        # We only take the first two features.
        samples = imputer.fit_transform(samples)

        cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)

        scores = cross_val_score(
            clf, samples, labels.values.ravel(), scoring='accuracy', cv=cv, n_jobs=-1)

        # Calculate accuracy
        accuracy = np.mean(scores)

        print("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s" % (accuracy, svc.n_support_, (time.time() -
                                                                                     start_time), svc.C, svc.kernel, svc.degree, svc.gamma, svc.coef0, svc.shrinking, svc.break_ties))

    return -accuracy

def differential_evolution_algorithm():
    bounds = [(1.0, 5.0), (1, 4), (0, 5), (1.0, 5.0), (0.0, 5.0), (0, 1), (0, 1)]

    return differential_evolution(
        func=svm,
        bounds=bounds,
        args=(),
        strategy='best1bin',
        maxiter=100,
        popsize=10,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        disp=False,
        polish=True,
        init='latinhypercube',
    )


# Quickly sample a training set while holding out 30% of the data for testing (evaluating) our classifier
samples_train, samples_test, labels_train, labels_test = read_and_split_dataset()

selected_features = feature_selection(samples_train, labels_train, None)

samples_train_selected_features = extract_features(selected_features, samples_train)
samples_test_selected_features = extract_features(selected_features, samples_test)

parameters = differential_evolution_algorithm()
svm(parameters.x, True)

# Calculate accuracy
# accuracy = clf.score(samples_test, labels_test)
# print('Mean accuracy: %s' % accuracy)

# # Some features to plot
# X = df.iloc[:,:2]
# y = labels

# # Plot Decision Region using mlxtend's awesome plotting function
# plot_decision_regions(
#     X=X.values,
#     y=y.values.ravel().astype(np.int64),
#     clf=clf,
#     legend=2
# )

# # Update plot object with X/Y axis labels and Figure Title
# plt.xlabel(X.columns[0], size=14)
# plt.ylabel(X.columns[1], size=14)
# plt.title('SVM Decision Region Boundary', size=16)
# plt.show()
