from differential_evolution import differential_evolution
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Crossover
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, precision_score, f1_score
from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearnex import patch_sklearn
from genetic_selection import GeneticSelectionCV
from thompson_sampling.bernoulli import BernoulliExperiment
import click
import numpy as np
import pandas as pd
import time

patch_sklearn(verbose=0)

class Easc:
    def __init__(self, evolutionaryAlgorithm, dataset, featureSelection, numberOfFeaturesToSelect, testSize, kernel, imputerStrategy, resultsFormat, printHeader, crossValidate, thompsonSampling):
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
        self.thompsonSamplingExperiment = None
        self.crossValidate = crossValidate
        self.thompsonSampling = thompsonSampling
        self.thompsonSamplingExperiment = None
        self.thompsonSamplingLastChoice = None
        self.thompsonSamplingChoices = {}

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
        if (int(self.crossValidate) == 1):
            return

        samples_train, samples_test, labels_train, labels_test = train_test_split(
            self.samples,
            self.labels,
            test_size=float(self.testSize),
        )

        self.samples_train = samples_train
        self.samples_test = samples_test
        self.labels_train = labels_train
        self.labels_test = labels_test

    def feature_selection(self):
        if self.featureSelection == None or self.featureSelection == 'None':
            return

        selector = self.get_feature_selector()

        if (int(self.crossValidate)):
            selector.fit_transform(self.samples, self.labels.values.ravel())
        else:
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
            1: SelectKBest(k=number_of_features_to_select),
            2: SelectPercentile(percentile=50),
            3: SelectFpr(),
            4: SelectFdr(),
            5: SelectFwe(),
            6: SequentialFeatureSelector(LinearSVC(dual=False), n_features_to_select=number_of_features_to_select, direction='forward'),
            7: SequentialFeatureSelector(LinearSVC(dual=False), n_features_to_select=number_of_features_to_select, direction='backward'),
            8: SelectFromModel(ExtraTreesClassifier(n_estimators=100), max_features=number_of_features_to_select),
            9: RFE(estimator=LinearSVC(dual=False), n_features_to_select=number_of_features_to_select, step=1),
            10: GeneticSelectionCV(
                LinearSVC(dual=False),
                cv=10,
                verbose=0,
                scoring="accuracy",
                max_features=number_of_features_to_select,
                n_population=30,
                crossover_proba=0.95,
                mutation_proba=0.01,
                n_generations=100,
                crossover_independent_proba=0.1,
                mutation_independent_proba=0.05,
                tournament_size=3,
                n_gen_no_change=5,
                caching=True,
                n_jobs=-1,
            )
        }

        if (int(self.featureSelection) in feature_selectors):
            return feature_selectors[int(self.featureSelection)]
        else:
            return None

    def thompson_sampling_init_experiment(self):
        if (int(self.thompsonSampling) == 0):
            return

        if self.evolutionaryAlgorithm == 'ga':
            arms = [
                'one_point',
                'two_point',
                'uniform',
                'uniform_window',
                'shuffle',
                'segment',
            ]

            self.thompsonSamplingExperiment = self.thompson_sampling_create_experiment(
                6,
                arms,
            )
        else:
            arms = [
                'best1bin',
                'best1exp',
                'rand1exp',
                'randtobest1exp',
                'currenttobest1exp',
                'best2exp',
                'rand2exp',
                'randtobest1bin',
                'currenttobest1bin',
                'best2bin',
                'rand2bin',
                'rand1bin'
            ]

            self.thompsonSamplingExperiment = self.thompson_sampling_create_experiment(
                12,
                arms
            )

        for arm in arms:
            self.thompsonSamplingChoices[arm] = 0

    def thompson_sampling_update_score(self, choosen_crossover, parent, child):
        if isinstance(parent, float):
            parent_fitness = parent * -1
        else:
            parent_fitness = self.classifier(parent) * -1

        if isinstance(child, float):
            child_fitness = child * -1
        else:
            child_fitness = self.classifier(child) * -1

        if (child_fitness > parent_fitness):
            self.thompsonSamplingExperiment.add_rewards([{"label": choosen_crossover, "reward": 1}])
        else:
            self.thompsonSamplingExperiment.add_rewards([{"label": choosen_crossover, "reward": 0}])

    def thompson_sampling_create_experiment(self, numberOfMachines, labels):
        return BernoulliExperiment(arms=numberOfMachines, labels=labels)

    def thompson_sampling_ga_crossover(self, x: np.ndarray, y: np.ndarray):
        crossovers = {
            'one_point': Crossover.one_point(),
            'two_point': Crossover.two_point(),
            'uniform': Crossover.uniform(),
            'uniform_window': Crossover.uniform_window(np.random.randint(1, x.size / 2)),
            'shuffle': Crossover.shuffle(),
            'segment': Crossover.segment(),
        }

        choosen_crossover = self.thompsonSamplingExperiment.choose_arm()
        self.thompsonSamplingLastChoice = choosen_crossover
        self.thompsonSamplingChoices[choosen_crossover] += 1

        child_x, child_y = crossovers[choosen_crossover](x, y)

        self.thompson_sampling_update_score(choosen_crossover, x, child_x)
        self.thompson_sampling_update_score(choosen_crossover, y, child_y)

        return child_x, child_y

    def thompson_sampling_de_mutation(self):
        choosen_mutation = self.thompsonSamplingExperiment.choose_arm()
        self.thompsonSamplingLastChoice = choosen_mutation
        self.thompsonSamplingChoices[choosen_mutation] += 1

        return choosen_mutation

    def thompson_sampling_de_mutation_callback(self, candidateFitness, trialFitness):
        self.thompson_sampling_update_score(self.thompsonSamplingLastChoice, candidateFitness, trialFitness)

    def thompson_sampling_get_most_choosen_arm(self):
        return max(self.thompsonSamplingChoices, key=self.thompsonSamplingChoices.get)

    def genetic_algorithm(self):
        if self.kernel == 'dynamic':
            varbound = np.array([
                [2e-10, 2e5],
                [1, 4],
                [2, 5],
                [2e-10, 2e5],
                [2e-10, 2e5]
            ])

            vartype = np.array(['real', 'int', 'int', 'real', 'real'])
        else:
            varbound = np.array([
                [2e-10, 2e5],
                [2, 5],
                [2e-10, 2e5],
                [2e-10, 2e5]
            ])

            vartype = np.array(['real', 'int', 'real', 'real'])

        if int(self.thompsonSampling) == 1:
            crossover = self.thompson_sampling_ga_crossover
        else:
            crossover = 'one_point'

        algorithm_parameters = {'max_num_iteration': 500,
                                'population_size': 30,
                                'mutation_probability': 0.01,
                                'elit_ratio': 0.00,
                                'crossover_probability': 0.95,
                                'parents_portion': 0.3,
                                'crossover_type': crossover,
                                'max_iteration_without_improv': 10,
                                'selection_type': 'roulette'}
        model = ga(
            algorithm_parameters=algorithm_parameters,
            function=self.classifier,
            dimension=len(varbound),
            variable_type_mixed=vartype,
            variable_boundaries=varbound,
            function_timeout=60,
        )

        model.run(
            no_plot=True,
            disable_progress_bar=True,
            disable_printing=True,
        )

        self.best_parameters = model.best_variable

    def differential_evolution(self):
        if self.kernel == 'dynamic':
            bounds = [
                (2e-10, 2e5),
                (1, 4),
                (2, 5),
                (2e-10, 2e5),
                (2e-10, 2e5),
            ]
        else:
            bounds = [
                (2e-10, 2e5),
                (2, 5),
                (2e-10, 2e5),
                (2e-10, 2e5),
            ]

        if int(self.thompsonSampling) == 1:
            mutationCallback = self.thompson_sampling_de_mutation_callback
            strategy = self.thompson_sampling_de_mutation
        else:
            mutationCallback = None
            strategy = 'rand1bin'

        self.best_parameters = differential_evolution(
            func=self.classifier,
            bounds=bounds,
            args=(),
            strategy=strategy,
            maxiter=500,
            popsize=20,
            tol=0.01,
            mutation=0.9314,
            mutation_callback=mutationCallback,
            recombination=0.6938,
            seed=None,
            disp=False,
            polish=True,
            init='random',
            maxfun=10000,
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
        if self.kernel == 'dynamic':
            svc = SVC(
                C=X[0],
                kernel=self.get_kernel(X[1]),
                degree=X[2],
                gamma=X[3],
                coef0=X[4],
                max_iter=100_000
            )
        else:
            svc = SVC(
                C=X[0],
                kernel=self.kernel,
                degree=X[1],
                gamma=X[2],
                coef0=X[3],
                max_iter=100_000
            )

        if (int(self.crossValidate) == 1):
            cv = KFold(n_splits=10)

            scores = cross_val_score(
                svc,
                self.samples,
                self.labels.values.ravel(),
                cv=cv,
                n_jobs=-1,
                verbose=0,
            )

            accuracy = np.mean(scores)
        else:
            svc.fit(self.samples_train, self.labels_train.values.ravel())

            accuracy = svc.score(self.samples_test, self.labels_test)

        return -accuracy

    def get_pos_label(self):
        mapping = {
            "winsconsin_569_32_normalizado": "B",
            "winsconsin_699_10_normalizado": "B",
            "coimbra_116_10_normalizado": "H",
            "winsconsin_198_34_normalizado": "N"
        }

        datasetName = self.dataset.split("/")[-1].split(".")[0]

        return mapping[datasetName]

    def calculate_metrics(self):
        if self.kernel == 'dynamic':
            self.best_svc = SVC(
                C=self.best_parameters[0],
                kernel=self.get_kernel(self.best_parameters[1]),
                degree=self.best_parameters[2],
                gamma=self.best_parameters[3],
                coef0=self.best_parameters[4],
                probability=True,
                max_iter=100_000
            )
        else:
            self.best_svc = SVC(
                C=self.best_parameters[0],
                kernel=self.kernel,
                degree=self.best_parameters[1],
                gamma=self.best_parameters[2],
                coef0=self.best_parameters[3],
                probability=True,
                max_iter=100_000
            )

        if (int(self.crossValidate) == 1):
            kf = KFold(n_splits=10)

            accuracy_scores = []
            sensitivity_scores = []
            specificity_scores = []
            recall_scores = []
            roc_aoc_scores = []
            confusion_matrixes = []
            precision_scores = []
            f1_scores = []
            g_mean_scores = []

            for train, test in kf.split(self.samples):
                X_train = np.array(self.samples)[train]
                X_test = np.array(self.samples)[test]
                y_train = np.array(self.labels)[train]
                y_test = np.array(self.labels)[test]

                self.best_svc.fit(X_train, y_train.ravel())

                accuracy = self.best_svc.score(X_test, y_test.ravel())

                predicted = self.best_svc.predict(X_test)

                sensitivity = sensitivity_score(
                    y_true=np.ravel(y_test),
                    y_pred=np.ravel(predicted),
                    pos_label=self.get_pos_label()
                )

                specificity = specificity_score(
                    y_true=np.ravel(y_test),
                    y_pred=np.ravel(predicted),
                    pos_label=self.get_pos_label()
                )

                recall = recall_score(
                    y_true=np.ravel(y_test),
                    y_pred=np.ravel(predicted),
                    pos_label=self.get_pos_label()
                )

                roc_aoc = roc_auc_score(
                    y_true=np.ravel(y_test),
                    y_score=self.best_svc.predict_proba(X_test)[:, 1],
                )

                c_matrix = confusion_matrix(
                    y_true=np.ravel(y_test),
                    y_pred=np.ravel(predicted),
                )

                precision = precision_score(
                    y_true=np.ravel(y_test),
                    y_pred=np.ravel(predicted),
                    pos_label=self.get_pos_label()
                )

                f1 = f1_score(
                    y_true=np.ravel(y_test),
                    y_pred=np.ravel(predicted),
                    pos_label=self.get_pos_label()
                )

                g_mean = geometric_mean_score(
                    y_true=np.ravel(y_test),
                    y_pred=np.ravel(predicted),
                    pos_label=self.get_pos_label()
                )

                accuracy_scores.append(accuracy)
                sensitivity_scores.append(sensitivity)
                specificity_scores.append(specificity)
                recall_scores.append(recall)
                roc_aoc_scores.append(roc_aoc)
                confusion_matrixes.append(c_matrix)
                precision_scores.append(precision)
                f1_scores.append(f1)
                g_mean_scores.append(g_mean)

            self.accuracy = np.mean(accuracy_scores)
            self.sensitivity_score = np.mean(sensitivity_scores)
            self.specificity_score = np.mean(specificity_scores)
            self.recall_score = np.mean(recall_scores)
            self.roc_aoc = np.mean(roc_aoc_scores)
            self.confusion_matrix = confusion_matrixes
            self.precision_score = np.mean(precision_scores)
            self.f1_score = np.mean(f1_scores)
            self.g_mean_score = np.mean(g_mean_scores)
        else:
            self.best_svc.fit(self.samples_train, self.labels_train.values.ravel())
            self.accuracy = self.best_svc.score(self.samples_test, self.labels_test)

            predicted = self.best_svc.predict(self.samples_test)

            self.sensitivity_score = sensitivity_score(
                y_true=np.ravel(self.labels_test),
                y_pred=np.ravel(predicted),
                pos_label=self.get_pos_label()
            )

            self.specificity_score = specificity_score(
                y_true=np.ravel(self.labels_test),
                y_pred=np.ravel(predicted),
                pos_label=self.get_pos_label()
            )

            self.recall_score = recall_score(
                y_true=np.ravel(self.labels_test),
                y_pred=np.ravel(predicted),
                pos_label=self.get_pos_label()
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
                pos_label=self.get_pos_label()
            )

            self.f1_score = f1_score(
                y_true=np.ravel(self.labels_test),
                y_pred=np.ravel(predicted),
                pos_label=self.get_pos_label()
            )

            self.g_mean_score = geometric_mean_score(
                y_true=np.ravel(self.labels_test),
                y_pred=np.ravel(predicted),
                pos_label=self.get_pos_label()
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

        if (int(self.thompsonSampling) == 1):
            if (self.results_format == 'txt'):
                if (self.print_header == '1'):
                    print("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s" % ("accuracy", "nsv", "time", "C", "kernel", "degree", "gamma", "coef0", "shrinking", "break_ties", "feature_mask", "number_of_selected_features", "sensitivity_score", "specificity_score", "recall_score", "roc_aoc", "confusion_matrix", "precision_score", "f1_score", "g_mean_score", "thompson_sampling_most_choosen_arm"))


                print("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s" % (self.accuracy, self.best_svc.n_support_, (self.end_time -
                                                                                        self.start_time), self.best_svc.C, self.best_svc.kernel, self.best_svc.degree, self.best_svc.gamma, self.best_svc.coef0, self.best_svc.shrinking, self.best_svc.break_ties, self.print_array(list(feature_mask)), sum(list(x == True for x in feature_mask)), self.sensitivity_score, self.specificity_score, self.recall_score, self.roc_aoc, self.print_matrix(self.confusion_matrix), self.precision_score, self.f1_score, self.g_mean_score, self.thompson_sampling_get_most_choosen_arm()))
            else:
                if (self.print_header == '1'):
                    print("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % ("accuracy", "nsv", "time", "C", "kernel", "degree", "gamma", "coef0", "shrinking", "break_ties", "feature_mask", "number_of_selected_features", "sensitivity_score", "specificity_score", "recall_score", "roc_aoc", "confusion_matrix", "precision_score", "f1_score", "g_mean_score", "thompson_sampling_most_choosen_arm"))

                print("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (self.accuracy, self.best_svc.n_support_, (self.end_time -
                                                                                     self.start_time), self.best_svc.C, self.best_svc.kernel, self.best_svc.degree, self.best_svc.gamma, self.best_svc.coef0, self.best_svc.shrinking, self.best_svc.break_ties, self.print_array(list(feature_mask)), sum(list(x == True for x in feature_mask)), self.sensitivity_score, self.specificity_score, self.recall_score, self.roc_aoc, self.print_matrix(self.confusion_matrix), self.precision_score, self.f1_score, self.g_mean_score, self.thompson_sampling_get_most_choosen_arm()))
        else:
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
        self.thompson_sampling_init_experiment()
        self.classify_dataset()
        self.calculate_metrics()
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
@click.option("--cross_validate", default='1', prompt="Cross validation (0 or 1)", help="Execute cross validation or not")
@click.option("--thompson_sampling", default='0', prompt="Thompson sampling (0 or 1)", help="Use thompson sampling or not")
def easc(evolutionary_algorithm, dataset, feature_selection, number_of_features_to_select, test_size, kernel, imputer_strategy, results_format, print_header, cross_validate, thompson_sampling):
    easc = Easc(evolutionary_algorithm, dataset, feature_selection, number_of_features_to_select, test_size, kernel, imputer_strategy, results_format, print_header, cross_validate, thompson_sampling)
    easc.run()

if __name__ == '__main__':
    easc()
