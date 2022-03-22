from scipy.optimize import differential_evolution
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from geneticalgorithm2 import geneticalgorithm2 as ga
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, precision_score, f1_score
from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from genetic_selection import GeneticSelectionCV
import click
import numpy as np
import pandas as pd
import time

class Easc:
    def __init__(self, evolutionaryAlgorithm, dataset, featureSelection, numberOfFeaturesToSelect, testSize, kernel, imputerStrategy, resultsFormat, printHeader):
        self.evolutionaryAlgorithm = evolutionaryAlgorithm
        self.dataset = dataset
        self.featureSelection = featureSelection
        self.numberOfFeaturesToSelect = numberOfFeaturesToSelect
        self.testSize = testSize
        self.kernel = kernel
        self.imputerStrategy = imputerStrategy
        self.sensitivity_score = None
        self.specificity_score = None
        self.recall_score = None
        self.roc_aoc = None
        self.confusion_matrix = None
        self.precision_score = None
        self.f1_score = None
        self.g_mean_score = None
        self.results_format = resultsFormat
        self.print_header = printHeader

    def load_dataset(self):
        # Read the csv data into a pandas data frame (df)
        df = pd.read_csv(
            filepath_or_buffer=self.dataset,
            header=0
        )

        self.samples = df.iloc[:, :-1]  # All columns except the last
        self.labels = df.iloc[:, -1:]  # The last column

    def normalize_dataset(self):
        # Replace all missing data (non numeric) with NaN (Not a Number)
        self.samples = self.samples.apply(lambda x: pd.to_numeric(x, errors='coerce'))

        # Fill mising values in the samples
        # We only take the first two features.
        self.samples = SimpleImputer(strategy=self.imputerStrategy).fit_transform(self.samples)

    def scale_dataset(self):
        self.samples = StandardScaler().fit_transform(self.samples)

    def split_dataset(self):
        samples_train, samples_test, labels_train, labels_test = train_test_split(
            self.samples,
            self.labels,
            test_size=self.testSize,
        )

        self.samples_train = samples_train
        self.samples_test = samples_test
        self.labels_train = labels_train
        self.labels_test = labels_test

    def feature_selection(self):
        if self.featureSelection == None or self.featureSelection == 'None':
            return

        selector = self.get_feature_selector()

        selector.fit_transform(self.samples_train, self.labels_train.values.ravel())

        self.feature_mask = selector._get_support_mask()

        indexes = []
        for i in range(len(self.feature_mask)):
            if self.feature_mask[i] == False:
                indexes.append(i)

        self.samples = np.delete(self.samples, indexes, axis=1)

    def get_feature_selector(self):
        number_of_features_to_select = None
        if self.numberOfFeaturesToSelect == 'all':
            number_of_features_to_select = int(self.samples.shape[1])
        elif self.numberOfFeaturesToSelect == 'half':
            number_of_features_to_select = int(round(self.samples.shape[1] / 2.0))
        else:
            number_of_features_to_select = int(self.numberOfFeaturesToSelect)

        feature_selectors = {
            1: VarianceThreshold(threshold=(.8 * (1 - .8))),
            2: SelectKBest(k=number_of_features_to_select),
            3: SelectPercentile(percentile=50),
            4: SelectFpr(),
            5: SelectFdr(),
            6: SelectFwe(),
            7: SequentialFeatureSelector(LinearSVC(dual=False), n_features_to_select=number_of_features_to_select, direction='forward'),
            8: SequentialFeatureSelector(LinearSVC(dual=False), n_features_to_select=number_of_features_to_select, direction='backward'),
            9: SelectFromModel(ExtraTreesClassifier(n_estimators=50), max_features=number_of_features_to_select),
            10: RFE(estimator=LinearSVC(dual=False), n_features_to_select=number_of_features_to_select, step=1),
            11: GeneticSelectionCV(
                LinearSVC(dual=False),
                cv=5,
                verbose=0,
                scoring="accuracy",
                max_features=5,
                n_population=50,
                crossover_proba=0.5,
                mutation_proba=0.2,
                n_generations=40,
                crossover_independent_proba=0.5,
                mutation_independent_proba=0.05,
                tournament_size=3,
                n_gen_no_change=2,
                caching=True,
                n_jobs=-1,
            )
        }

        if (int(self.featureSelection) in feature_selectors):
            return feature_selectors[int(self.featureSelection)]
        else:
            return None

    def genetic_algorithm(self):
        varbound = np.array([[1.0, 5.0], [1, 4], [0, 5], [
                            1.0, 5.0], [0.0, 5.0], [0, 1], [0, 1]])
        vartype = np.array(['real', 'int', 'int',
                        'real', 'real', 'int', 'int'])

        algorithm_parameters = {'max_num_iteration': 100,
                                'population_size': 10,
                                'mutation_probability': 0.1,
                                'elit_ratio': 0.01,
                                'crossover_probability': 0.5,
                                'parents_portion': 0.3,
                                'crossover_type': 'uniform',
                                'max_iteration_without_improv': 5}
        model = ga(
            algorithm_parameters=algorithm_parameters,
            function=self.classifier,
            dimension=len(varbound),
            variable_type_mixed=vartype,
            variable_boundaries=varbound,
            function_timeout=60
        )

        model.run(
            no_plot=True,
            disable_progress_bar=True,
            disable_printing=True,
        )

        self.best_parameters = model.best_variable

    def differential_evolution(self):
        bounds = [(1.0, 5.0), (1, 4), (0, 5), (1.0, 5.0), (0.0, 5.0), (0, 1), (0, 1)]

        self.best_parameters = differential_evolution(
            func=self.classifier,
            bounds=bounds,
            args=(),
            strategy='rand1bin',
            maxiter=100,
            popsize=10,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=None,
            disp=False,
            polish=True,
            init='random',
        ).x

    def classify_dataset(self):
        if (self.evolutionaryAlgorithm == 'ga'):
            self.genetic_algorithm()
        elif (self.evolutionaryAlgorithm == 'de'):
            self.differential_evolution()
        else:
            raise Exception('Invalid evolutionary algorithm!')

    def get_kernel(self, kernel_id):
        if (self.kernel != 'dynamic'):
            return self.kernel

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
            raise Exception('Invalid kernel!')

    def classifier(self, X):
        svc = SVC(
            C=X[0],
            kernel=self.get_kernel(X[1]),
            degree=X[2],
            gamma=X[3],
            coef0=X[4],
            shrinking=X[5],
            break_ties=X[6],
        )

        svc.fit(self.samples_train, self.labels_train.values.ravel())

        accuracy = svc.score(self.samples_test, self.labels_test)

        return -accuracy

    def calculate_accuracy(self):
        self.best_svc = SVC(
            C=self.best_parameters[0],
            kernel=self.get_kernel(self.best_parameters[1]),
            degree=self.best_parameters[2],
            gamma=self.best_parameters[3],
            coef0=self.best_parameters[4],
            shrinking=self.best_parameters[5],
            break_ties=self.best_parameters[6],
            probability=True
        )

        cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)

        self.best_svc.fit(self.samples_train, self.labels_train.values.ravel())

        self.scores = cross_validate(
            self.best_svc,
            self.samples,
            self.labels.values.ravel(),
            scoring='accuracy',
            cv=cv,
            n_jobs=-1,
        )

        predicted = self.best_svc.predict(self.samples_test)

        self.accuracy = np.mean(self.scores['test_score'])

        self.sensitivity_score = sensitivity_score(
            y_true=np.ravel(self.labels_test),
            y_pred=np.ravel(predicted),
            pos_label='B'
        )

        self.specificity_score = specificity_score(
            y_true=np.ravel(self.labels_test),
            y_pred=np.ravel(predicted),
            pos_label='B'
        )

        self.recall_score = recall_score(
            y_true=np.ravel(self.labels_test),
            y_pred=np.ravel(predicted),
            pos_label='B'
        )

        self.roc_aoc = roc_auc_score(
            y_true=np.ravel(self.labels_test),
            y_score=self.best_svc.predict_proba(self.samples_test)[:, 1],
        )

        self.confusion_matrix = confusion_matrix(
            y_true=np.ravel(self.labels_test),
            y_pred=np.ravel(predicted),
        )

        self.precision_score = precision_score(
            y_true=np.ravel(self.labels_test),
            y_pred=np.ravel(predicted),
            pos_label='B'
        )

        self.f1_score = f1_score(
            y_true=np.ravel(self.labels_test),
            y_pred=np.ravel(predicted),
            pos_label='B'
        )

        self.g_mean_score = geometric_mean_score(
            y_true=np.ravel(self.labels_test),
            y_pred=np.ravel(predicted),
            pos_label='B'
        )

    def print_array(self, array):
        string = '[ '
        index = 0

        for x in range(len(array)):
            if (array[x] == True):
                string += (str(index) + ' ')
            index += 1

        string += ']'

        return string

    def print_matrix(self, matrix):
        string = ''

        for i in range(len(matrix)):
            string += '['
            numbers = ''

            for j in range(len(matrix[i])):
                    numbers += (str(matrix[i][j]) + ' ')

            string += numbers.strip(' ') + ']'

        return string

    def print(self):
        feature_mask = []
        if (hasattr(self, 'feature_mask')):
            feature_mask = self.feature_mask

        if (self.results_format == 'txt'):
            if (self.print_header == '1'):
                print("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s" % ("accuracy", "nsv", "time", "C", "kernel", "degree", "gamma", "coef0", "shrinking", "break_ties", "feature_mask", "number_of_selected_features", "sensitivity_score", "specificity_score", "recall_score", "roc_aoc", "confusion_matrix", "precision_score", "f1_score", "g_mean_score"))


            print("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s" % (self.accuracy, self.best_svc.n_support_, (self.end_time -
                                                                                     self.start_time), self.best_svc.C, self.best_svc.kernel, self.best_svc.degree, self.best_svc.gamma, self.best_svc.coef0, self.best_svc.shrinking, self.best_svc.break_ties, self.print_array(list(feature_mask)), sum(list(x == True for x in feature_mask)), self.sensitivity_score, self.specificity_score, self.recall_score, self.roc_aoc, self.print_matrix(self.confusion_matrix), self.precision_score, self.f1_score, self.g_mean_score))
        else:
            if (self.print_header == '1'):
                print("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % ("accuracy", "nsv", "time", "C", "kernel", "degree", "gamma", "coef0", "shrinking", "break_ties", "feature_mask", "number_of_selected_features", "sensitivity_score", "specificity_score", "recall_score", "roc_aoc", "confusion_matrix", "precision_score", "f1_score", "g_mean_score"))

            print("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (self.accuracy, self.best_svc.n_support_, (self.end_time -
                                                                                     self.start_time), self.best_svc.C, self.best_svc.kernel, self.best_svc.degree, self.best_svc.gamma, self.best_svc.coef0, self.best_svc.shrinking, self.best_svc.break_ties, self.print_array(list(feature_mask)), sum(list(x == True for x in feature_mask)), self.sensitivity_score, self.specificity_score, self.recall_score, self.roc_aoc, self.print_matrix(self.confusion_matrix), self.precision_score, self.f1_score, self.g_mean_score))

    def run(self):
        self.start_time = time.time()
        self.load_dataset()
        self.normalize_dataset()
        self.scale_dataset()
        self.split_dataset()
        self.feature_selection()
        self.classify_dataset()
        self.calculate_accuracy()
        self.end_time = time.time()
        self.print()

@click.command()
@click.option("--evolutionary_algorithm", default="ga", prompt="Evolutionary algorithm (ga or de)", help="The evolutionary algorithm to use.")
@click.option("--dataset", prompt="Path to dataset", help="The path to dataset to be used (csv).")
@click.option("--feature_selection", default="None", prompt="Feature selection (1-10 or None)", help="The feature selection method to use.")
@click.option("--number_of_features_to_select", default="half", prompt="Number of features to select, only applicable if feature selection is enabled", help="The number of features to select.")
@click.option("--test_size", default=0.3, prompt="Test size (0.1-1.0)", help="The percentual of the dataset do use in testing (0.0 to 1.0).")
@click.option("--kernel", default='dynamic', prompt="Kernel (linear, poly, rbf, sigmoid or dynamic)", help="The kernel to use in the SVM, If None, the kernel will be choosen automatically.")
@click.option("--imputer_strategy", default='most_frequent', prompt="Imputer strategy", help="The strategy to use in the Imputer to fill missing values.")
@click.option("--results_format", default='txt', prompt="Output format", help="The format to output the results (csv or txt)")
@click.option("--print_header", default='1', prompt="Print header", help="Print the header of the results (0 or 1)")
def easc(evolutionary_algorithm, dataset, feature_selection, number_of_features_to_select, test_size, kernel, imputer_strategy, results_format, print_header):
    easc = Easc(evolutionary_algorithm, dataset, feature_selection, number_of_features_to_select, test_size, kernel, imputer_strategy, results_format, print_header)
    easc.run()

if __name__ == '__main__':
    easc()
