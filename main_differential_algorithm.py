from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time


progress = []
start_time = time.time()

def f(X, printer=False):
    # Read the csv data into a pandas data frame (df)
    df = pd.read_csv(
        filepath_or_buffer='./datasets/winsconsin_699_10_normalizado.csv',
        header=0
    )

    samples = df.iloc[:,:-1] # All columns except the last
    labels = df.iloc[:,-1:] # The last column

    # Replace all missing data (non numeric) with NaN (Not a Number)
    samples = samples.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # Create a inputer to fill missing values
    imputer = SimpleImputer(strategy='most_frequent')

    # Fill mising values in the samples
    # We only take the first two features.
    samples = imputer.fit_transform(samples)

    # Quickly sample a training set while holding out 30% of the data for testing (evaluating) our classifier
    samples_train, samples_test, labels_train, labels_test = train_test_split(
        samples,
        labels,
        test_size=0.3,
    )

    svc = SVC(
        C=X[0],
        kernel='poly',
        degree=X[1],
        gamma=X[2],
        coef0=X[3],
        shrinking=X[4],
        break_ties=X[5],
    );

    # Creating a pipeline with the scaler and the classifier
    clf = make_pipeline(
        StandardScaler(),
        svc,
    )

    # Fit the classifier with the training data
    clf.fit(
        samples_train,
        labels_train.values.ravel()  # Transform a column vector to 1d array
    )

    # Calculate accuracy
    accuracy = clf.score(samples_test, labels_test)

    if (printer):
        print("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s" % (accuracy, svc.n_support_, (time.time() -
                                                                                 start_time), svc.C, svc.kernel, svc.degree, svc.gamma, svc.coef0, svc.shrinking, svc.break_ties))

    return -accuracy

def save_value(x, convergence):
    progress.append(f(x))

bounds = [(1.0, 5.0), (0, 5), (1.0, 5.0), (0.0, 5.0), (0, 1), (0, 1)]
result = differential_evolution(
    func=f,
    bounds=bounds,
    args=(),
    strategy='best1bin',
    maxiter=100,
    popsize=10,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=None,
    callback=save_value,
    disp=False,
    polish=True,
    init='latinhypercube',
)

f(result.x, True)

# Plot
# plt.plot(progress)
# plt.ylabel('Mean Accuracy')
# plt.xlabel('Progress')
# plt.show()
