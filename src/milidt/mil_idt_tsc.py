import copy
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import stats
from sklearn.metrics import accuracy_score
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier, HoeffdingTreeClassifier, ExtremelyFastDecisionTreeClassifier


class MilIdtTsc:
    """Multi-Instance Learning based Incremental Decision Trees for Time-Series Classification.

    Parameters
    ----------
    base_model
        Decision tree learner to perform the classification.
        - 'HT' - Hoeffding Tree (Very Fast Decision Tree)
        - 'HATT' - Hoeffding Anytime Tree (Extremely Fast Decision Tree)
        - 'HAT - Hoeffding Adaptive Tree

    inst_len
        Length of the instances to split the sequence.
    inst_stride
        Stride, in time steps, between the start of each instance.
    k
        Number of consecutive instances to search the signature of the sequence.
    grace_period
        Number of time steps a leaf of the decision tree should observe between split attempts.
    split_confidence
        Significance level to calculate the Hoeffding bound. Values closer to zero imply longer split decision delays.
    seed
        Random seed for reproducibility.
    """

    def __init__(self, base_model: str, inst_len=10, inst_stride=1, k=3, grace_period=2000, split_confidence=1e-7, seed=0):
        self.base_model = base_model
        self.inst_len = inst_len
        self.inst_stride = inst_stride
        self.k = k
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

    def train(self, train_data: np.ndarray, max_it=50, patience=5, reset_model=False, verbose=1) -> dict:
        """Train the model given a set of time series.

        Parameters
        ----------
        train_data
            2D NumPy array with first column as the sequence identifier and last column as the class label of the sequence.
        max_it
            Maximum number of iterations for searching the best instances in each bag.
        patience
            Number of iterations without improving accuracy on train before stopping the search. Employed criterion: accuracy.
        reset_model
            If activated, the decision tree is reset between iterations to forget the previous training with other instances.
        verbose
            Level of verbosity: 0 silent, 1 verbose.

        Returns
        -------
        A dictionary with the results of the training:
        - 'model' - Learned decision tree.
        - 'y_true' - Actual class labels of the time series.
        - 'y_pred' - Predicted class labels of the time series.
        - 'selection' - Indexes of the selected instances in each bag corresponding to each sequence.
        - 'history' - Performance reached over training iterations.
        - 'exec_time' - Time of execution for training in seconds.
        """
        init_time = time.perf_counter()
        if verbose > 0:
            print("Training")

        # First fitting with all the instances
        if verbose > 0:
            print(f"Iteration 0 of {max_it}")

        units, idxs = np.unique(train_data[:, 0], return_index=True)
        data_per_unit = np.split(train_data[:, 1:], idxs[1:])
        y_true = train_data[idxs, -1]

        self._fit(train_data)
        y_pred = self._evaluation(data_per_unit)
        acc = accuracy_score(y_true, y_pred)
        best_acc = acc

        if verbose > 0:
            print(f"Current accuracy: {acc}")

        # Data to return
        result = {
            'model': copy.deepcopy(self.classifier),
            'y_true': y_true,
            'y_pred': y_pred,
            'selection': None,
            'history': [acc]
        }

        # Searching best k consecutive instances for every bag
        curr_it = 0
        curr_pat = patience
        while curr_it < max_it and curr_pat > 0:
            if verbose > 0:
                print(f"Iteration {curr_it + 1} of {max_it}")
            selection = self._bestk(data_per_unit, y_true, idxs)
            if reset_model:
                self.classifier.reset()
            self._fit(train_data, selection=selection)
            y_pred = self._evaluation(data_per_unit)
            acc = accuracy_score(y_true, y_pred)

            if verbose > 0:
                print(f"Current accuracy: {acc}")

            result['history'].append(acc)
            if acc > best_acc:
                result['model'] = copy.deepcopy(self.classifier)
                result['y_pred'] = y_pred
                result['selection'] = selection
                best_acc = acc
                curr_pat = patience
            else:
                curr_pat -= 1
            curr_it += 1

        end_time = time.perf_counter()
        result['exec_time'] = end_time - init_time

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
        - 'selection' - Indexes of the selected instances in each bag corresponding to each sequence.
        - 'exec_time' - Time of execution for testing in seconds.
        """
        init_time = time.perf_counter()
        if verbose > 0:
            print("Testing")

        units, idxs = np.unique(test_data[:, 0], return_index=True)
        data_per_unit = np.split(test_data[:, 1:], idxs[1:])
        y_true = test_data[idxs, -1]

        # Testing performance
        y_pred = self._evaluation(data_per_unit)

        # Finding K most significant instances
        selection = self._bestk(data_per_unit, y_pred, idxs)

        end_time = time.perf_counter()
        result = {
            'y_true': y_true,
            'y_pred': y_pred,
            'selection': selection,
            'exec_time': end_time - init_time,
        }
        if verbose > 0:
            print(f"Accuracy on test: {accuracy_score(y_true, y_pred)}")

        return result

    def _fit(self, data: np.ndarray, selection=None) -> None:
        """Split each sequence in bag of instances to train the model. The selection parameter indicates the instances to consider; if it is set to None, all the instances of the bags are used.
        """
        units, idxs = np.unique(data[:, 0], return_index=True)
        if selection is not None:
            _, train_idxs = np.unique(data[selection, 0], return_index=True)
            train_per_unit = np.split(data[selection, 1:], train_idxs[1:])
        else:
            train_per_unit = np.split(data[:, 1:], idxs[1:])

        for seq in train_per_unit:
            actual_len = min(self.inst_len, len(seq))
            roll_win = sliding_window_view(seq, window_shape=actual_len, axis=0)[::self.inst_stride]
            for instance in roll_win:
                instance_x = instance.transpose()[:, :-1]
                instance_y = instance.transpose()[:, -1]
                self.classifier.partial_fit(instance_x, instance_y)

    def _evaluation(self, data_per_unit: list) -> np.ndarray:
        """Generate the predicted class labels for each instance based on the standard MIL assumption.
        """
        y_pred = np.zeros(len(data_per_unit), dtype=int)
        for i, seq in enumerate(data_per_unit):
            actual_len = min(self.inst_len, len(seq))
            roll_win = sliding_window_view(seq, window_shape=actual_len, axis=0)[::self.inst_stride]
            for instance in roll_win:
                instance_x = instance.transpose()[:, :-1]
                prediction = stats.mode(self.classifier.predict(instance_x))[0]
                y_pred[i] = max(y_pred[i], prediction)
        return y_pred

    def _bestk(self, data_per_unit: list, y_ref: np.ndarray, idxs: np.ndarray) -> list:
        """Select the best k instances of every bag based on the maximization of the likelihood of the prediction.
        """
        selection = list()
        for i, seq in enumerate(data_per_unit):
            seq[:, -1] = y_ref[i]  # Test likelihood according to reference label (actual if train, predicted if test)
            actual_len = min(self.inst_len, len(seq))
            roll_win = sliding_window_view(seq, window_shape=actual_len, axis=0)[::self.inst_stride]
            instances_likelihood = np.zeros(len(roll_win))
            actual_k = min(self.k, len(roll_win))
            for j, instance in enumerate(roll_win):
                instance = instance.transpose()
                instances_likelihood[j] = np.sum(np.apply_along_axis(self._likelihood, 1, instance))
            roll_likelihood = sliding_window_view(instances_likelihood, window_shape=actual_k, axis=0)
            max_win = np.argmax(np.sum(roll_likelihood, axis=1))
            start = idxs[i] + (max_win * self.inst_stride)
            end = idxs[i] + (max_win + actual_k - 1) * self.inst_stride + actual_len
            selection.extend(np.r_[start:end])
        return selection

    def _likelihood(self, working_cycle: np.array) -> float:
        likelihood = self.classifier.get_votes_for_instance(working_cycle[:-1])
        try:
            return likelihood[working_cycle[-1]]
        except:
            return 0
