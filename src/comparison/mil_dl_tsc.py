import functools
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import stats
from sklearn.metrics import accuracy_score
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Bidirectional, Conv1D, Conv2D, Dense, Dropout, Flatten, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing.timeseries import timeseries_dataset_from_array


class MilDlTsc:
    """Multi-Instance Learning based Deep Learning for Time-Series Classification.

    Parameters
    ----------
    base_model
        Deep neural network to perform the classification.
        - 'LSTM' - Long short-term memory model
        - 'BidLSTM' - Bidirectional LSTM
        - 'CNN' - Convolutional neural network
        - 'CNNLSTM' - LSTM with previous processing using CNN
        - 'Dense' - Deep multi-layer perceptron
    n_inputs
        Number of attributes in time series (needed for initialize the deep models)
    n_outputs
        Number of classes to learn (needed for initialize the deep models)
    learning_rate
        Step size in the optimization function during the training of the deep model.
    batch_size
        Number of instances per gradient update.
    inst_len
        Length of the instances to split the sequence.
    inst_stride
        Stride, in time steps, between the start of each instance.
    k
        Number of consecutive instances to search the signature of the sequence.
    seed
        Random seed for reproducibility.
    """

    def __init__(self, base_model: str, n_inputs: int, n_outputs: int, learning_rate=1e-3, batch_size=128, inst_len=10, inst_stride=1, k=3, seed=0):
        self.base_model = base_model
        self.inst_len = inst_len
        self.inst_stride = inst_stride
        self.k = k
        self.batch_size = batch_size
        self.seed = seed

        random.seed(self.seed)
        tf.random.set_seed(self.seed)

        if self.base_model == 'LSTM':
            self.classifier = self.lstm_model(n_inputs, n_outputs)
        elif self.base_model == 'BidLSTM':
            self.classifier = self.bidlstm_model(n_inputs, n_outputs)
        elif self.base_model == 'CNN':
            self.classifier = self.cnn_model(n_inputs, n_outputs, self.inst_len)
        elif self.base_model == 'CNNLSTM':
            self.classifier = self.lstmcnn_model(n_inputs, n_outputs)
        elif self.base_model == 'Dense':
            self.classifier = self.dense_model(n_inputs, n_outputs)
        else:
            raise ValueError("ERROR: invalid base classifier ({self.base_model})")

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.classifier.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, train_data: np.ndarray, epochs=1, max_it=50, patience=10, verbose=1) -> dict:
        """Train the model given a set of time series.

        Parameters
        ----------
        train_data
            2D NumPy array with first column as the sequence identifier and last column as the class label of the sequence.
        epochs
            Number of epochs to perform during network fitting in each training iteration.
        max_it
            Maximum number of iterations for searching the best instances in each bag.
        patience
            Number of iterations without improving accuracy on train before stopping the search. Employed criterion: accuracy.
        verbose
            Level of verbosity: 0 silent, 1 verbose.

        Returns
        -------
        A dictionary with the results of the training:
        - 'model' - Trained deep model.
        - 'y_true' - Actual class labels of the time series.
        - 'y_pred' - Predicted class labels of the time series.
        - 'selection' - Indexes of the selected instances in each bag corresponding to each sequence.
        - 'history' - Performance reached over training iterations.
        - 'exec_time' - Time of execution for training in seconds.
        """
        init_time = time.perf_counter()
        if verbose == 1:
            print("Training")
            self.classifier.summary()
            print(f"Iteration 0 of {max_it}")

        units, idxs = np.unique(train_data[:, 0], return_index=True)
        data_per_unit = np.split(train_data[:, 1:], idxs[1:])
        y_true = train_data[idxs, -1]

        # First fitting with all the instances
        self._fit(train_data, epochs, verbose=verbose)
        y_pred = self._evaluation(data_per_unit, verbose)
        acc = accuracy_score(y_true, y_pred)
        best_acc = acc

        if verbose == 1:
            print(f"Current accuracy: {acc}")

        # Data to return
        result = {
            'y_true': y_true,
            'y_pred': y_pred,
            'selection': None,
            'history': [acc]
        }
        saved_weights = self.classifier.get_weights()

        # Searching best k consecutive instances for every bag
        curr_it = 0
        curr_pat = patience
        while curr_it < max_it and curr_pat > 0:
            if verbose == 1:
                print(f"Iteration {curr_it + 1} of {max_it}")
            selection = self._bestk(data_per_unit, y_true, idxs, verbose)
            self._fit(train_data, epochs, selection, verbose)
            y_pred = self._evaluation(data_per_unit, verbose)
            acc = accuracy_score(y_true, y_pred)

            if verbose == 1:
                print(f"Current accuracy: {acc}")

            result['history'].append(acc)
            if acc > best_acc:
                saved_weights = self.classifier.get_weights()
                result['y_pred'] = y_pred
                result['selection'] = selection
                best_acc = acc
                curr_pat = patience
            else:
                curr_pat -= 1
            curr_it += 1

        end_time = time.perf_counter()
        result['exec_time'] = end_time - init_time
        self.classifier.set_weights(saved_weights)
        result['model'] = self.classifier

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
        if verbose == 1:
            print("Testing")

        units, idxs = np.unique(test_data[:, 0], return_index=True)
        data_per_unit = np.split(test_data[:, 1:], idxs[1:])
        y_true = test_data[idxs, -1]

        # Testing performance
        y_pred = self._evaluation(data_per_unit, verbose)

        # Finding k most significant instances
        selection = self._bestk(data_per_unit, y_pred, idxs, verbose)

        end_time = time.perf_counter()
        result = {
            'y_true': y_true,
            'y_pred': y_pred,
            'selection': selection,
            'exec_time': end_time - init_time,
        }
        if verbose == 1:
            print(f"Accuracy on test: {accuracy_score(y_true, y_pred)}")

        return result

    def _fit(self, data: np.ndarray, epochs: int, selection=None, verbose=0) -> None:
        """Split each sequence in bag of instances to train the model. The selection parameter indicates the instances to consider; if it is set to None, all the instances of the bags are used.
        """
        units, idxs = np.unique(data[:, 0], return_index=True)
        if selection is not None:
            _, train_idxs = np.unique(data[selection, 0], return_index=True)
            train_per_unit = np.split(data[selection, 1:], train_idxs[1:])
        else:
            train_per_unit = np.split(data[:, 1:], idxs[1:])

        if self.base_model != 'Dense':
            dldata_train = list()
            dldata_val = list()
            for i, seq in enumerate(train_per_unit):
                for_train = random.random() > 0.2
                actual_len = min(self.inst_len, len(seq))
                roll_win = timeseries_dataset_from_array(data=seq[:, :-1], targets=seq[:, -1], sequence_length=actual_len, sequence_stride=self.inst_stride, batch_size=self.batch_size)
                if for_train:
                    dldata_train.append(roll_win)
                else:
                    dldata_val.append(roll_win)
            dldata_train = functools.reduce(lambda acc, other: acc.concatenate(other), dldata_train)
            dldata_val = functools.reduce(lambda acc, other: acc.concatenate(other), dldata_val)
            self.classifier.fit(dldata_train, validation_data=dldata_val, epochs=epochs, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=verbose)
        else:
            dldata = list()
            for i, seq in enumerate(train_per_unit):
                actual_len = min(self.inst_len, len(seq))
                roll_win = sliding_window_view(seq, window_shape=actual_len, axis=0)[::self.inst_stride]
                for instance in roll_win:
                    dldata.append(instance.transpose())
            dldata = np.concatenate(dldata, axis=0)
            self.classifier.fit(dldata[:, :-1], dldata[:, -1], epochs=epochs, batch_size=self.batch_size, validation_split=0.2, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=verbose)

    def _evaluation(self, data_per_unit: list, verbose=0) -> np.ndarray:
        """Generate the predicted class labels for each instance based on the standard MIL assumption.
        """
        y_pred = np.zeros(len(data_per_unit), dtype=int)
        dldata = list()
        bags_info = list()  # Keeps for later [#instances, actual window length] of each bag
        if self.base_model != 'Dense':
            for i, seq in enumerate(data_per_unit):
                actual_len = min(self.inst_len, len(seq))
                roll_win_x = timeseries_dataset_from_array(data=seq[:, :-1], targets=None, sequence_length=actual_len, sequence_stride=self.inst_stride, batch_size=self.batch_size)
                dldata.append(roll_win_x)
                bags_info.append([0, actual_len])
                for batch in roll_win_x:
                    bags_info[-1][0] += batch.shape[0]
            dldata = functools.reduce(lambda acc, other: acc.concatenate(other), dldata)
        else:
            for i, seq in enumerate(data_per_unit):
                actual_len = min(self.inst_len, len(seq))
                roll_win_x = sliding_window_view(seq[:, :-1], window_shape=actual_len, axis=0)[::self.inst_stride]
                bags_info.append([roll_win_x.shape[0], actual_len])
                for instance in roll_win_x:
                    instance = instance.transpose()
                    dldata.append(instance)
            dldata = np.concatenate(dldata, axis=0)

        probs = self.classifier.predict(dldata, batch_size=self.batch_size, verbose=verbose)

        ref_idx = 0
        if self.base_model != 'Dense':
            for i in range(len(data_per_unit)):
                pred_bag = np.argmax(probs[ref_idx:ref_idx+bags_info[i][0]], axis=1)
                ref_idx += bags_info[i][0]
                y_pred[i] = max(pred_bag)
        else:
            for i in range(len(data_per_unit)):
                pred_bag = list()
                for _ in range(bags_info[i][0]):
                    pred_bag.append(stats.mode(np.argmax(probs[ref_idx:ref_idx+bags_info[i][1], :], axis=1))[0])
                    ref_idx += bags_info[i][1]
                y_pred[i] = max(pred_bag)
        return y_pred

    def _bestk(self, data_per_unit, y_ref, idxs: np.ndarray, verbose=0) -> list:
        """Select the best k instances of every bag based on the maximization of the likelihood of the prediction.
        """
        selection = list()

        dldata = list()
        bags_info = list()  # Keeps for later [#instances, actual window length] of each bag
        roll_likelihood = list()

        loss = self.SaveLoss(reduction=tf.keras.losses.Reduction.NONE)
        self.classifier.compile(optimizer=self.classifier.optimizer, loss=loss.compute, run_eagerly=True)

        if self.base_model != 'Dense':
            for i, seq in enumerate(data_per_unit):
                seq[:, -1] = y_ref[i]  # Test likelihood according to reference label (actual if train, predicted if test)
                actual_len = min(self.inst_len, len(seq))
                roll_win = timeseries_dataset_from_array(data=seq[:, :-1], targets=seq[:, -1], sequence_length=actual_len, sequence_stride=self.inst_stride, batch_size=self.batch_size)
                dldata.append(roll_win)
                bags_info.append([0, actual_len])
                for batch_x, _ in roll_win:
                    bags_info[-1][0] += batch_x.shape[0]
            dldata = functools.reduce(lambda acc, other: acc.concatenate(other), dldata)

            self.classifier.evaluate(dldata, verbose=verbose)

            ref_idx = 0
            for i in range(len(data_per_unit)):
                actual_k = min(self.k, bags_info[i][0])
                roll_likelihood.append(sliding_window_view(loss.losses[ref_idx:ref_idx+bags_info[i][0]], window_shape=actual_k, axis=0))
                ref_idx += bags_info[i][0]
        else:
            # Dense model does not process the entire sequence, but receive each step at a time.
            for i, seq in enumerate(data_per_unit):
                seq[:, -1] = y_ref[i]  # Test likelihood according to reference label (actual if train, predicted if test)
                actual_len = min(self.inst_len, len(seq))
                roll_win = sliding_window_view(seq, window_shape=actual_len, axis=0)[::self.inst_stride]
                bags_info.append([roll_win.shape[0], roll_win.shape[2]])
                for j, instance in enumerate(roll_win):
                    instance = instance.transpose()
                    dldata.append(instance)
            dldata = np.concatenate(dldata, axis=0)

            self.classifier.evaluate(dldata[:, :-1], dldata[:, -1], batch_size=self.batch_size, verbose=verbose)

            ref_idx = 0
            for i in range(len(data_per_unit)):
                actual_k = min(self.k, bags_info[i][0])
                instances_likelihood = list()
                for _ in range(bags_info[i][0]):
                    instances_likelihood.append(np.sum(loss.losses[ref_idx:ref_idx+bags_info[i][1]]))
                    ref_idx += bags_info[i][1]
                roll_likelihood.append(sliding_window_view(instances_likelihood, window_shape=actual_k, axis=0))

        for i, idx in enumerate(idxs):
            max_win = np.argmin(np.sum(roll_likelihood[i], axis=1))
            actual_k = min(self.k, bags_info[i][0])
            start = idx + (max_win * self.inst_stride)
            end = idx + (max_win + actual_k - 1) * self.inst_stride + bags_info[i][1]
            selection.extend(np.r_[start:end])

        self.classifier.compile(optimizer=self.classifier.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return selection

    @staticmethod
    def lstm_model(n_inputs, n_outputs) -> Model:
        input_layer = Input((None, n_inputs))  # Samples, steps, attributes
        lstm = LSTM(32, return_sequences=False)(input_layer)
        lstm = Dense(12, activation='relu')(lstm)
        lstm = Dropout(.2)(lstm)
        lstm = Dense(n_outputs, activation='softmax')(lstm)
        return Model(inputs=input_layer, outputs=lstm)

    @staticmethod
    def bidlstm_model(n_inputs, n_outputs) -> Model:
        input_layer = Input((None, n_inputs))
        bidlstm = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
        bidlstm = Bidirectional(LSTM(32, return_sequences=False))(bidlstm)
        bidlstm = Dense(12, activation='relu')(bidlstm)
        bidlstm = Dropout(.2)(bidlstm)
        bidlstm = Dense(n_outputs, activation='softmax')(bidlstm)
        return Model(inputs=input_layer, outputs=bidlstm)

    @staticmethod
    def cnn_model(n_inputs, n_outputs, seq_len) -> Model:
        input_layer = Input((seq_len, n_inputs, 1))
        cnn = Conv2D(64, 3, activation='relu')(input_layer)
        cnn = Conv2D(12, 2, activation='relu')(cnn)
        cnn = Flatten()(cnn)
        cnn = Dense(6, activation='relu')(cnn)
        cnn = Dropout(.2)(cnn)
        cnn = Dense(n_outputs, activation='softmax')(cnn)
        return Model(inputs=input_layer, outputs=cnn)

    @staticmethod
    def lstmcnn_model(n_inputs, n_outputs) -> Model:
        input_layer = Input((None, n_inputs))
        lstmcnn = Conv1D(12, 3, activation='relu')(input_layer)
        lstmcnn = LSTM(64, return_sequences=True)(lstmcnn)
        lstmcnn = LSTM(32, return_sequences=False)(lstmcnn)
        lstmcnn = Dense(16, activation='relu')(lstmcnn)
        lstmcnn = Dropout(.2)(lstmcnn)
        lstmcnn = Dense(n_outputs, activation='softmax')(lstmcnn)
        return Model(inputs=input_layer, outputs=lstmcnn)

    @staticmethod
    def dense_model(n_inputs, n_outputs) -> Model:
        input_layer = Input((n_inputs,))
        dense = Dense(32, activation='relu')(input_layer)
        dense = Dense(64, activation='relu')(dense)
        dense = Dense(16, activation='relu')(dense)
        dense = Dropout(.2)(dense)
        dense = Dense(n_outputs, activation='softmax')(dense)
        return Model(inputs=input_layer, outputs=dense)

    class SaveLoss:
        def __init__(self, reduction):
            self.scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=reduction)
            self.losses = list()

        def compute(self, y_true, y_pred):
            result = self.scce(y_true, y_pred)
            self.losses.extend(result.numpy())
            return result


if __name__ == '__main__':
    data_path = "turbofan_categorical"
    data_key = "FD001"
    cols = ['unit'] + [f"s{i}" for i in range(1, 22)] + ['HS']#['unit', 'DE', 'FE', 'BA', 'RPM', 'HS']
    n_inputs, n_outputs = 21, 3
    train_df = pd.read_csv(f"{data_path}/train_{data_key}.csv")
    test_df = pd.read_csv(f"{data_path}/test_{data_key}.csv")

    train_data = train_df[cols].to_numpy()
    test_data = test_df[cols].to_numpy()

    mi_dl = MilDlTsc('CNN', n_inputs=n_inputs, n_outputs=n_outputs, inst_len=10, inst_stride=3, batch_size=1)
    train_result = mi_dl.train(train_data, epochs=2)
    test_result = mi_dl.test(test_data)
