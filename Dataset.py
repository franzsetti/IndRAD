import os
import pandas as pd
import random
import numpy as np


class Dataset:
    DEFAULT_COLUMNS = [f'{key}_{i}' for key in ['position', 'velocity', 'effort'] for i in range(7)]
    DATA_BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    TRAIN_DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset_train')
    TEST_NOMINAL_DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset_test_nominal')
    TEST_ANOMALY_DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset_test_anomaly')
    CORRUPTION_TYPES = ['freeze_zero', 'freeze_last_value', 'spike', 'step']

    def __init__(self, name, dataset=None, window_size=None, scaler=None):
        """
        ****************************************************************
        * Class Dataset                                                *
        ****************************************************************
        * name          : name of the dataset
        * dataset       : dataset
        * window_size   : set the window size for the elaboration
        * scaler        : the scaler to use for the normalization
        """
        self.name = name
        self.window_size = window_size
        self.scaler = scaler
        self.is_windowed = False
        self.is_normalized = False

        self.dataset = dataset
        self.dataset_processed = self.dataset.copy() if self.dataset is not None else None

        self.errors = None
        self.anomalies = {}

    def load(self, path, with_window=False):
        """
        Load dataset from file system
        * path          : path of the file
        * with_window   : the dataset to load has sliding windows
        """
        filename = f"{self.name}_{self.window_size}.csv" if (
                    self.window_size and self.window_size > 1 and with_window) else f"{self.name}.csv"

        if with_window and self.window_size > 1:
            self.dataset_processed = pd.read_csv(os.path.join(path, filename), sep=";", dtype='float64')
            self.dataset = self.remove_sliding_window(window_size=self.window_size, in_place=False)
            self.is_windowed = True
        else:
            self.dataset = pd.read_csv(os.path.join(path, filename), sep=";", dtype='float64')
            self.dataset_processed = self.dataset.copy()

    def apply_moving_average(self, step):
        if step and step > 1:
            self.dataset = self.dataset.rolling(step).mean().dropna().reset_index(drop=True)
            self.dataset_processed = self.dataset.copy()
            self.is_windowed = False
            self.is_normalized = False

    def add_sliding_window(self, window_size=None, save_path=None, in_place=True):
        """
        Add sliding temporal window
        * window_size   : the size of the window
        * save_path     : path where to save the dataset with window, if not None
        * in_place      : add sliding window in self.dataset_processed
        ===========================================================================
        Return:
            * dataset with sliding windows
        """
        if window_size:
            self.window_size = window_size

        if self.dataset is None:
            raise Exception("** Dataset required! **")

        dataset_window = None

        if self.window_size and self.window_size > 1:
            while self.dataset.shape[0] % self.window_size != 0:
                self.dataset = self.dataset[:-1]

            dataset_window = self.dataset.copy()

            dataset_window = dataset_window.to_numpy().flatten().reshape(
                dataset_window.shape[0] // self.window_size,
                dataset_window.shape[1] * self.window_size)
            dataset_window = pd.DataFrame(dataset_window)

            if save_path:
                dataset_window.to_csv(os.path.join(save_path, f"{self.name}_{self.window_size}.csv"), sep=";",
                                      index=False)
            if in_place:
                self.dataset_processed = dataset_window.copy()
                self.is_windowed = True
        return dataset_window

    def remove_sliding_window(self, window_size=None, in_place=True):
        """
        Remove sliding temporal window
        * window_size   : the size of the window
        * in_place      : remove sliding window from self.dataset_processed
        """
        if window_size:
            self.window_size = window_size

        dataset_no_window = Dataset.dataset_remove_sliding_window(dataset=self.dataset_processed, window_size=self.window_size)

        if in_place:
            self.dataset_processed = dataset_no_window.copy()
            self.is_windowed = False

        return dataset_no_window

    def normalize(self, scaler=None, in_place=True):
        """
        Normalize the dataset with the scaler
        * scaler : the scaler to utilize for the normalization, if None use self.scaler
        """
        if scaler is not None:
            self.scaler = scaler
        if self.scaler is None:
            raise Exception("** Scaler required! **")

        dataset_normalized = pd.DataFrame(self.scaler.transform(self.dataset_processed))
        if in_place:
            self.dataset_processed = dataset_normalized.copy()
        self.is_normalized = True
        return dataset_normalized

    def normalize_inverse(self, in_place=True):
        """
        Inverse normalization of the dataset processed
        """
        if self.scaler is None:
            raise Exception("** Scaler required! **")

        dataset_inverse_normalized = Dataset.dataset_normalize_inverse(dataset=self.dataset_processed, scaler=self.scaler)
        if in_place:
            self.dataset_processed = dataset_inverse_normalized.copy()
        self.is_normalized = False
        return dataset_inverse_normalized

    def pca(self, model):
        """
        Apply PCA transformation
        * model : pca model
        """
        self.dataset_processed = pd.DataFrame(model.transform(self.dataset_processed))

    def pca_inverse(self, model):
        """
        PCA Inverse transformation
        * model : pca model used for transformation
        ========================
        Return:
            * reconstruction errors (without window)
        """
        self.dataset_processed = pd.DataFrame(model.inverse_transform(self.dataset_processed))

        if self.scaler is not None:
            _dataset_processed = Dataset.dataset_normalize_inverse(dataset=self.dataset_processed, scaler=self.scaler)
        else:
            _dataset_processed = self.dataset_processed.copy()

        if self.is_windowed and self.window_size > 1:
            _dataset_processed = Dataset.dataset_remove_sliding_window(dataset=_dataset_processed, window_size=self.window_size)
        else:
            _dataset_processed = _dataset_processed.copy()

        self.errors = (self.dataset - _dataset_processed).to_numpy()

        return self.errors

    def corrupt(self, corruption_type='random', save_path=None):
        """
        - freeze
        - freeze_last_value
        - spike
        """
        if corruption_type == 'random' or corruption_type not in Dataset.CORRUPTION_TYPES:
            corruption_type = random.choice(Dataset.CORRUPTION_TYPES)
        self.name += "_" + corruption_type

        eval(f"self.{corruption_type}")()
        if save_path:
            pass

    def freeze_zero(self):
        if self.dataset is None:
            raise Exception("** Dataset required! **")

        col = random.choice(self.dataset.columns)
        if col not in self.anomalies.keys():
            self.anomalies[col] = np.array([])

        self.dataset.loc[len(self.dataset) // 2:, col] = 0.0
        self.anomalies[col] = np.unique(
            np.append(self.anomalies[col], np.arange(len(self.dataset) // 2, len(self.dataset))))
        self.dataset_processed = self.dataset.copy()

        self.is_windowed = False
        self.is_normalized = False

        return self.anomalies

    def freeze_last_value(self):
        if self.dataset is None:
            raise Exception("** Dataset required! **")

        col = random.choice(self.dataset.columns)
        if col not in self.anomalies.keys():
            self.anomalies[col] = np.array([])

        self.dataset.loc[len(self.dataset) // 2:, col] = self.dataset.loc[
            len(self.dataset) // 2, col]
        self.anomalies[col] = np.unique(
            np.append(self.anomalies[col], np.arange(len(self.dataset) // 2, len(self.dataset))))
        self.dataset_processed = self.dataset.copy()

        self.is_windowed = False
        self.is_normalized = False

        return self.anomalies

    def spike(self):
        if self.dataset is None:
            raise Exception("** Dataset required! **")

        col = random.choice(self.dataset.columns)
        if col not in self.anomalies.keys():
            self.anomalies[col] = np.array([])
        for i in range(0, 1, 1):
            error = 500
            #error = random.randint(int(max(self.dataset[col]) * 3), int(5 * max(self.dataset[col])))
            index = random.randint(0, len(self.dataset) - 1)

            self.dataset.loc[index, col] += error
            self.anomalies[col] = np.unique(np.append(self.anomalies[col], index))
        self.dataset_processed = self.dataset.copy()

        self.is_windowed = False
        self.is_normalized = False

        return self.anomalies

    def step(self):
        if self.dataset is None:
            raise Exception("** Dataset required! **")

        col = random.choice(self.dataset.columns)
        if col not in self.anomalies.keys():
            self.anomalies[col] = np.array([])

        error = random.randint(int(max(self.dataset[col])*5+1), int(20 * max(self.dataset[col])+1))
        self.dataset.loc[len(self.dataset) // 2:, col] += error
        self.anomalies[col] = np.unique(
            np.append(self.anomalies[col], np.arange(len(self.dataset) // 2, len(self.dataset))))
        self.dataset_processed = self.dataset.copy()

        self.is_windowed = False
        self.is_normalized = False

        return self.anomalies

    @staticmethod
    def dataset_normalize_inverse(dataset, scaler):
        return pd.DataFrame(scaler.inverse_transform(dataset))

    @staticmethod
    def dataset_remove_sliding_window(dataset, window_size):
        return pd.DataFrame(
            dataset.to_numpy().flatten().reshape(dataset.shape[0] * window_size, len(Dataset.DEFAULT_COLUMNS)),
            columns=Dataset.DEFAULT_COLUMNS)

    @staticmethod
    def save_dataset(dataset, path, name):
        if not os.path.isdir(path):
            os.mkdir(path)
        if dataset:
            dataset.to_csv(os.path.join(path, f"{name}.csv"))

    @staticmethod
    def create_dataset(trajectories,
                       output_path,
                       data_base_path=None,
                       save_single_dataset=True,
                       save_full_dataset=True,
                       full_dataset_name="full_dataset"):
        data_base_path = data_base_path if data_base_path is not None else Dataset.DATA_BASE_PATH

        full_dataset = []

        for trajectory in trajectories:
            if os.path.isdir(os.path.join(data_base_path, trajectory)):
                position = velocity = effort = pd.DataFrame()

                csv_position_path = os.path.join(data_base_path, trajectory, 'positions.csv')
                csv_velocity_path = os.path.join(data_base_path, trajectory, 'velocities.csv')
                csv_torque_path = os.path.join(data_base_path, trajectory, 'torques.csv')

                if os.path.isfile(csv_position_path):
                    position = pd.read_csv(csv_position_path, sep=";", header=None,
                                           names=[f'position_{i}' for i in range(7)])
                if os.path.isfile(csv_velocity_path):
                    velocity = pd.read_csv(csv_velocity_path, sep=";", header=None,
                                           names=[f'velocity_{i}' for i in range(7)])
                if os.path.isfile(csv_torque_path):
                    effort = pd.read_csv(csv_torque_path, sep=";", header=None, names=[f'effort_{i}' for i in range(7)])

                dataset = pd.concat([position, velocity, effort], axis=1)

                if save_full_dataset:
                    full_dataset.append(dataset)

                if not os.path.isdir(output_path):
                    os.mkdir(output_path)

                if save_single_dataset:
                    dataset.to_csv(os.path.join(output_path, f"{trajectory}.csv"), sep=";", index=False)

        if save_full_dataset:
            full_dataset = pd.concat(full_dataset, axis=0)
            full_dataset.to_csv(os.path.join(output_path, f'{full_dataset_name}.csv'), sep=";", index=False)
