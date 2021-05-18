from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from genetic_selection import GeneticSelectionCV
import numpy as np
import pandas as pd
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


def read_and_split_dataset(test_size=0.3):
    # Read the csv data into a pandas data frame (df)
    df = pd.read_csv(
        filepath_or_buffer='./datasets/winsconsin_569_32_normalizado.csv',
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
    estimator = SVC(kernel='poly')

    selector = GeneticSelectionCV(
        estimator,
        cv=5,
        verbose=1,
        scoring="accuracy",
        max_features=max_features,
        n_population=500,
        crossover_proba=0.5,
        mutation_proba=0.2,
        n_generations=50,
        crossover_independent_proba=0.5,
        mutation_independent_proba=0.05,
        tournament_size=3,
        n_gen_no_change=10,
        caching=True,
        n_jobs=-1
    )

    # Creating a pipeline with the scaler and the classifier
    clf = make_pipeline(
        StandardScaler(),
        selector,
    )

    # Fit the classifier with the training data
    clf.fit(
        samples_train,
        labels_train.values.ravel()  # Transform a column vector to 1d array
    )

    return selector.support_


def extract_features(selected_features, samples):
    indexes = []

    for i in range(len(selected_features)):
        if selected_features[i] == True:
            print(i)
            indexes.append(i)

    return np.delete(samples, indexes, axis=1)

def svm(X):
    # Replace all missing data (non numeric) with NaN (Not a Number)
    samples_train_df = pd.DataFrame.from_records(samples_train_selected_features).apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # Create a inputer to fill missing values
    imputer = SimpleImputer(strategy='most_frequent')

    # Fill mising values in the samples
    samples_fit = imputer.fit_transform(samples_train_df)

    # Creating a pipeline with the scaler and the classifier
    clf = make_pipeline(
        StandardScaler(),
        SVC(
            C=X[0],
            kernel='poly',
            degree=X[1],
            gamma=X[2],
            coef0=X[3],
            shrinking=X[4],
            break_ties=X[5],
        ),
    )

    # Fit the classifier with the training data
    clf.fit(
        samples_fit,
        labels_train.values.ravel()  # Transform a column vector to 1d array
    )

    return -clf.score(samples_test_selected_features, labels_test)


def differential_evolution_algorithm():
    bounds = [(1.0, 50.0), (0, 5), (1.0, 50.0), (0.0, 50.0), (0, 1), (0, 1)]

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
        disp=True,
        polish=True,
        init='latinhypercube',
    )


# Quickly sample a training set while holding out 30% of the data for testing (evaluating) our classifier
samples_train, samples_test, labels_train, labels_test = read_and_split_dataset()

selected_features = feature_selection(samples_train, labels_train, 5)

samples_train_selected_features = extract_features(selected_features, samples_train)
samples_test_selected_features = extract_features(selected_features, samples_test)

parameters = differential_evolution_algorithm()
print('Parameters:')
print(parameters)

accuracy = svm(parameters.x)
print('Mean accuracy:')
print(accuracy)

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
