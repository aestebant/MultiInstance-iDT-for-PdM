import copy
import time

import numpy as np

from collections import defaultdict

import pandas as pd
from scipy.stats import mode, skew, kurtosis
from sklearn import neighbors, svm, tree, ensemble, naive_bayes
from sklearn.metrics import accuracy_score
from skmultiflow.trees import HoeffdingTreeClassifier


class SilFeTsc:
    """Single-Instance Learning based on Feature Extraction for Time-series Classification.

    Parameters
    ----------
    base_model
        Machine learning model to perform the classification.
        - 'kNN' - k Nearest Neighbors
        - 'SVM' - Support Vector Machines
        - 'NB' - Gaussian Naïve Bayes
        - 'DT' - Decision Tree
        - 'RF' - Random Forest
        - 'HT' - Hoeffding Tree
    advanced_features
        If activated, skewness of sample and kurtosis are also included in the features set.
    features_per_attribute
        It controls whether if the features are computed for each attribute separately or all together.
    """
    
    def __init__(self, base_model: str, advanced_features=True, features_per_attribute=True):
        self.base_model = base_model
        if self.base_model == 'kNN':
            self.classifier = neighbors.KNeighborsClassifier(n_neighbors=5)
        elif self.base_model == 'SVM':
            self.classifier = svm.SVC(kernel='poly', max_iter=10000)
        elif self.base_model == 'NB':
            self.classifier = naive_bayes.GaussianNB()
        elif self.base_model == 'DT':
            self.classifier = tree.DecisionTreeClassifier()
        elif self.base_model == 'RF':
            self.classifier = ensemble.RandomForestClassifier()
        elif self.base_model == 'HT':
            self.classifier = HoeffdingTreeClassifier()
        else:
            raise ValueError(f"ERROR: invalid base classifier ({self.base_model})")
        self.advanced_features = advanced_features
        self.features_per_attribute = features_per_attribute

    def train(self, train_data: np.ndarray, verbose=1):
        """ Train the model given a set of time series.

        Parameters
        ----------
        train_data
            2D NumPy array with first column as the sequence identifier and last column as the class label of the sequence.
        verbose
            Level of verbosity: 0 silent, 1 verbose.

        Returns
        -------
        A dictionary with the results of the training:
        - 'model' - Trained machine learning model.
        - 'y_true' - Actual class labels of the time series.
        - 'y_pred' - Predicted class labels of the time series.
        - 'exec_time' - Time of execution for training in seconds.
        """
        init_time = time.perf_counter()
        if verbose == 1:
            print("Training")

        units, idxs = np.unique(train_data[:, 0], return_index=True)
        data_per_unit = np.split(train_data[:, 1:-1], idxs[1:])
        y_true = train_data[idxs, -1]

        # Feature extraction
        fe_data = self._featureextraction(data_per_unit)
        x_train = fe_data.to_numpy()

        # Train
        self.classifier.fit(x_train, y_true)
        # Train performance
        y_pred = self.classifier.predict(x_train)

        if verbose == 1:
            acc = accuracy_score(y_true, y_pred)
            print(f"Current accuracy: {acc}")

        end_time = time.perf_counter()
        # Data to return
        result = {
            'model': copy.deepcopy(self.classifier),
            'y_true': y_true,
            'y_pred': y_pred,
            'exec_time': end_time - init_time,
        }
        return result

    def test(self, test_data: np.ndarray, verbose=1) -> dict:
        """Testing the model on a give set of time series.

        Parameters
        ----------
        test_data
            2D NumPy array with first column as the sequence identifier and last column as the class label of the sequence.
        verbose
            Level of verbosity: 0 silent, 1 verbose.

        Returns
        -------
        A dictionary with the results of the testing:
        - 'y_true' - Actual class labels of the time series.
        - 'y_pred' - Predicted class labels of the time series.
        - 'exec_time' - Time of execution for testing in seconds.
        """
        init_time = time.perf_counter()
        if verbose == 1:
            print("Testing")

        units, idxs = np.unique(test_data[:, 0], return_index=True)
        data_per_unit = np.split(test_data[:, 1:-1], idxs[1:])
        y_true = test_data[idxs, -1]

        # Feature extraction
        fe_data = self._featureextraction(data_per_unit)
        x_test = fe_data.to_numpy()
        # Test performance
        y_pred = self.classifier.predict(x_test)

        if verbose == 1:
            acc = accuracy_score(y_true, y_pred)
            print(f"Current accuracy: {acc}")

        end_time = time.perf_counter()
        result = {
            'y_true': y_true,
            'y_pred': y_pred,
            'exec_time': end_time - init_time
        }
        return result

    def _featureextraction(self, data_per_unit) -> pd.DataFrame:
        result = defaultdict(lambda: list())

        if self.features_per_attribute:
            for seq in data_per_unit:
                fe_data = defaultdict(lambda: list())
                fe_data['sum'] = list(np.sum(seq, axis=0))
                fe_data['mean'] = list(np.mean(seq, axis=0))
                fe_data['median'] = list(np.median(seq, axis=0))
                fe_data['min'] = list(np.min(seq, axis=0))
                fe_data['max'] = list(np.max(seq, axis=0))
                fe_data['mode'] = list(mode(seq, axis=0)[0][0])
                fe_data['std'] = list(np.std(seq, axis=0))
                if self.advanced_features:
                    fe_data['skew'] = list(skew(seq, axis=0))
                    fe_data['kurtosis'] = list(kurtosis(seq, axis=0))
                aux = pd.DataFrame(fe_data)
                for i, row in aux.iterrows():
                    result[f'sum{i}'].append(row['sum'])
                    result[f'mean{i}'].append(row['mean'])
                    result[f'median{i}'].append(row['median'])
                    result[f'min{i}'].append(row['min'])
                    result[f'max{i}'].append(row['max'])
                    result[f'mode{i}'].append(row['mode'])
                    result[f'std{i}'].append(row['std'])
                    if self.advanced_features:
                        result[f'skew{i}'].append(row['skew'])
                        result[f'kurtosis{i}'].append(row['kurtosis'])
        else:
            for seq in data_per_unit:
                result['sum'].append(np.sum(seq, axis=None))
                result['mean'].append(np.mean(seq, axis=None))
                result['median'].append(np.median(seq, axis=None))
                result['min'].append(np.min(seq, axis=None))
                result['max'].append(np.max(seq, axis=None))
                result['mode'].append(mode(seq, axis=None)[0][0])
                result['std'].append(np.std(seq, axis=None))
                if self.advanced_features:
                    result['skew'].append(skew(seq, axis=None))
                    result['kurtosis'].append(kurtosis(seq, axis=None))

        return pd.DataFrame(result)


if __name__ == '__main__':
    data_path = "bearing_processed"
    data_key = "drive-innerrace-1200-2"
    cols = ['unit', 'DE', 'FE', 'BA', 'RPM', 'HS']#['unit'] + [f"s{i}" for i in range(1, 22)] + ['HS']
    train_df = pd.read_csv(f"{data_path}/train_{data_key}.csv")
    test_df = pd.read_csv(f"{data_path}/test_{data_key}.csv")

    train_data = train_df[cols].to_numpy()
    test_data = test_df[cols].to_numpy()

    simple_fe_tsc = SilFeTsc('NB')
    train_result = simple_fe_tsc.train(train_data)
    test_result = simple_fe_tsc.test(test_data)
