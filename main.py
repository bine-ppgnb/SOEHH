from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import time

start_time = time.time()

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
    C=1.0,
    kernel='poly',
    degree=3,
    gamma='scale',
    coef0=0.0,
    shrinking=True,
    break_ties=False,
);

# Creating a pipeline with the scaler and the classifier
clf = make_pipeline(
    StandardScaler(),
    svc
)

# Fit the classifier with the training data
clf.fit(
    samples_train,
    labels_train.values.ravel()  # Transform a column vector to 1d array
)

# Calculate accuracy
accuracy = clf.score(samples_test, labels_test)
# print('Accuracy: %s' % accuracy)

# Number of support vectors
# print('Number of support vectors: %s' % svc.n_support_)

# Time
# print("Time: %s seconds" % (time.time() - start_time))

print("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s" % (accuracy, svc.n_support_, (time.time() - start_time), svc.C, svc.kernel, svc.degree, svc.gamma, svc.coef0, svc.shrinking, svc.break_ties))

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
