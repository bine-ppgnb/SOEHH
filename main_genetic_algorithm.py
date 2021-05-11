from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga

def f(X):
    # Read the csv data into a pandas data frame (df)
    df = pd.read_csv(
        filepath_or_buffer='./datasets/winsconsin_569_32_normalizado.csv',
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
        samples_train,
        labels_train.values.ravel()  # Transform a column vector to 1d array
    )

    # Calculate accuracy
    accuracy = clf.score(samples_test, labels_test)
    print('Mean accuracy: %s' % accuracy)

    return -accuracy


varbound = np.array([[1.0, 50.0], [0, 5], [1.0, 50.0],
                    [0.0, 50.0], [0, 1], [0, 1]])
vartype = np.array([['real'], ['int'], ['real'], ['real'], ['int'], ['int']])

algorithm_parameters = {'max_num_iteration': 100,
                        'population_size': 10,
                        'mutation_probability': 0.1,
                        'elit_ratio': 0.01,
                        'crossover_probability': 0.5,
                        'parents_portion': 0.3,
                        'crossover_type': 'uniform',
                        'max_iteration_without_improv': None}
model = ga(
    algorithm_parameters=algorithm_parameters,
    function=f,
    dimension=6,
    variable_type_mixed=vartype,
    variable_boundaries=varbound,
)

model.run()

print(model.param)

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
