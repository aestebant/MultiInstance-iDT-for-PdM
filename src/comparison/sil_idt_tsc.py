import copy
import time

import numpy as np
from scipy.stats import stats
from sklearn.metrics import accuracy_score
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier, HoeffdingTreeClassifier, ExtremelyFastDecisionTreeClassifier


class SilIdtTsc:
    """Single-Instance Learning based Incremental Decision Trees for Time-Series Classification.

    Parameters
    ----------
    base_model
        Decision tree learner to perform the classification.
        - 'HT' - Hoeffding Tree (Very Fast Decision Tree)
        - 'HATT' - Hoeffding Anytime Tree (Extremely Fast Decision Tree)
        - 'HAT - Hoeffding Adaptive Tree
    grace_period
        Number of time steps a leaf of the decision tree should observe between split attempts.
    split_confidence
        Significance level to calculate the Hoeffding bound. Values closer to zero imply longer split decision delays.
    seed
        Random seed for reproducibility.
    """

    def __init__(self, base_model: str, grace_period=200, split_confidence=1e-7, seed=0):
        self.base_model = base_model
        self.seed = seed
        if self.base_model == 'HAT':
            self.classifier = HoeffdingAdaptiveTreeClassifier(grace_period=grace_period, split_confidence=split_confidence, random_state=seed)
        elif base_model == 'HT':
            self.classifier = HoeffdingTreeClassifier(grace_period=grace_period, split_confidence=split_confidence)
        elif base_model == 'HATT':
            self.classifier = ExtremelyFastDecisionTreeClassifier(grace_period=grace_period, split_confidence=split_confidence)
        else:
            raise ValueError(f"ERROR: invalid base classifier ({self.base_model})")

    def set_classifier(self, classifier):
        if isinstance(classifier, HoeffdingAdaptiveTreeClassifier):
            self.classifier = classifier
        elif isinstance(classifier, HoeffdingTreeClassifier):
            self.classifier = classifier
        elif isinstance(classifier, ExtremelyFastDecisionTreeClassifier):
            self.classifier = classifier
        else:
            raise ValueError(f'ERROR: invalid base classifier ({classifier.__class__.__name__})')

    def train(self, train_data: np.ndarray, verbose=1) -> dict:
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
        - 'model' - Learned decision tree.
        - 'y_true' - Actual class labels of the time series.
        - 'y_pred' - Predicted class labels of the time series.
        - 'exec_time' - Time of execution for training in seconds.
        """
        init_time = time.perf_counter()
        if verbose == 1:
            print("Training")

        units, idxs = np.unique(train_data[:, 0], return_index=True)
        data_per_unit = np.split(train_data[:, 1:], idxs[1:])
        y_true = train_data[idxs, -1]

        # Train
        for i, u in enumerate(units):
            sequence_x = data_per_unit[i][:, :-1]
            sequence_y = data_per_unit[i][:, -1]
            self.classifier.partial_fit(sequence_x, sequence_y)

        # Train performance
        y_pred = self._evaluation(data_per_unit)

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
        data_per_unit = np.split(test_data[:, 1:], idxs[1:])
        y_true = test_data[idxs, -1]

        # Test performance
        y_pred = self._evaluation(data_per_unit)

        if verbose == 1:
            acc = accuracy_score(y_true, y_pred)
            print(f"Current accuracy: {acc}")

        end_time = time.perf_counter()
        # Data to return
        result = {
            'y_true': y_true,
            'y_pred': y_pred,
            'exec_time': end_time - init_time,
        }
        return result

    def _evaluation(self, data_per_unit: list) -> np.ndarray:
        """Generate the predicted class labels for each instance using the mode of the sequence.
        """
        y_pred = np.zeros(len(data_per_unit), dtype=int)
        for i, seq in enumerate(data_per_unit):
            sequence_x = seq[:, :-1]  # Remove class label in last column
            y_pred[i] = stats.mode(self.classifier.predict(sequence_x))[0]
        return y_pred
