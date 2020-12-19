import os
from enum import Enum

import numpy as np
import torch
from torch.nn import Linear, Sequential, ReLU

# from utility.Explorer import Explorer

# to remove MacOS specific warnings for MKL
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.random.seed(2)
torch.manual_seed(1)


class Optimizer(Enum):
    ADAM = 1
    SGD = 2
    RMSPROP = 3


class NN:
    """A neural network model for timeseries forecasting

    Arguments:
        None (add what you need)
    """
    model = None

    # All hyperparamters of this model
    hyperparameters = {
        "hl_1_units": None,
        "hl_2_units": None,
        "learning_rate": None,
        "momentum": None,
        "nesterov": None,
        "optimizer": None
    }

    # leaned parameters of the model
    learned_paramters = {
        "min_val_loss": None,
        "min_val_loss_epoch": None
    }

    def __init__(self, hl_1_units=500, hl_2_units=200, learning_rate=0.001, momentum=0.9, nesterov=False, optimizer=Optimizer.ADAM):
        """
        The default arguments have been provided based on the best model found during experimentation phase.
        Args:
            hl_1_units:
            hl_2_units:
            learning_rate:
            momentum:
            nesterov:
            optimizer:
        """
        self.hyperparameters = {
            "hl_1_units": hl_1_units,
            "hl_2_units": hl_2_units,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "nesterov": nesterov,
            "optimizer": optimizer
        }
        self.learned_paramters = {
            "min_val_loss": np.inf,
            "min_val_loss_epoch": -1
        }
        self.model = self.__get_model_class(hl_1_units, hl_2_units)
        print(self.model)

    def __get_model_class(self, hl_1_units, hl_2_units):
        if hl_2_units == 0 or hl_2_units is None:
            return Sequential(
                Linear(300, hl_1_units),
                ReLU(),
                Linear(hl_1_units, 100)
            )
        return Sequential(
            Linear(300, hl_1_units),
            ReLU(),
            Linear(hl_1_units, hl_2_units),
            ReLU(),
            Linear(hl_2_units, 100)
        )

    def objective(self, X, y):
        """Compute objective function.

        Arguments:
            X (numpy ndarray, shape = (samples, 300)):
                Observed portion of timeseries for each data case.
            y  (numpy ndarray, shape = (samples, 100)):
                Portion of time series to predict for each data case

        Returns:
            float: Mean squared error objective value. Mean is taken
            over all dimensions of all data cases.
        """
        y_pred = self.predict(X)
        return self.calculate_loss(y, y_pred)
        # return np.mean(np.square(y - y_pred))

    def calculate_loss(self, y, y_pred):
        """

        Args:
            y  (numpy ndarray, shape = (samples, 100)):
                Portion of time series to predict for each data case
            y_pred  (numpy ndarray, shape = (samples, 100)):
                Portion of time series predicted for each data case

        Returns:
            float: Mean squared error objective value. Mean is taken
            over all dimensions of all data cases.
        """
        return np.mean(np.square(y - y_pred))

    def predict(self, X):
        """Forecast time series values.

        Arguments:
            X (numpy ndarray, shape = (samples, 300)):
                Observed portion of timeseries for each data case.

        Returns:
            y  (numpy ndarray, shape = (samples,100)):
                Predicted portion of time series for each data case.
        """
        x_in = torch.tensor(self.__reshape_input_X(X)).float()
        y_out = self.model(x_in)
        y = y_out.detach().numpy()
        return y

    def __reshape_input_X(self, X):
        """
        Private method to convert input shape which is suitable for this model
        Args:
            X (numpy ndarray, shape = (samples, 300)):
                Input samples
        Returns:
            X (numpy ndarray, shape = (samples, 1, 300)):
                Input samples which are reshaped
        """
        # num_samples = X.shape[0]
        # num_features = X.shape[1]
        # num_channels = 1
        # return np.reshape(X, (num_samples, num_channels, num_features))
        return X

    def print_parameters(self):
        parameters = list(self.model.parameters())
        print("num parameters ", len(parameters))
        # print(self.model.state_dict())
        # print("parameters list:", parameters)

    def fit_with_validation_data_check(self, X_train, y_train, X_val, y_val, epochs=300):
        """Train the model using the given training data.

        Arguments:
            y_val (numpy ndarray, shape = (samples, 100)):
                Validation data timeseries prediction
            X_val (numpy ndarray, shape = (samples, 300)):
                Validation data timeseries
            X_train (numpy ndarray, shape = (samples, 300)):
                Observed portion of timeseries for each data case
            y_train  (numpy ndarray, shape = (samples, 100)):
                Portion of time series to predict for each data case
            step (float):
                step size to use
            epochs (int):
                number of epochs of training
        """
        # input tensor
        x_train_in = torch.tensor(self.__reshape_input_X(X_train)).float()
        x_val_in = torch.tensor(self.__reshape_input_X(X_val)).float()

        # output tensor
        y_train_out = torch.tensor(y_train).float()
        y_val_out = torch.tensor(y_val).float()

        loss_fun = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters["learning_rate"])
        loss_train = np.inf
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # forward pass
            y_train_pred = self.model(x_train_in)
            y_val_pred = self.model(x_val_in)
            if epoch == 0:
                print("shape of y_train_pred", y_train_pred.shape)

            # compute the loss
            loss_train = loss_fun(y_train_pred, y_train_out)
            loss_val = loss_fun(y_val_pred, y_val_out)
            train_losses.append(loss_train)
            val_losses.append(loss_val)
            if epoch % 10 == 0:
                print("Train and Validation losses after epoch ", epoch, loss_train.item(), loss_val.item())

            # save best validation loss
            if loss_val.item() < self.learned_paramters["min_val_loss"]:
                self.learned_paramters["min_val_loss"] = loss_val.item()
                self.learned_paramters["min_val_loss_epoch"] = epoch + 1

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        print("Final loss of model: ", loss_train.item())
        print("Best validation loss {} at epoch {} for model {}".format(
            self.learned_paramters["min_val_loss"],
            self.learned_paramters["min_val_loss_epoch"],
            self.get_model_unique_name()
        ))
        # explorer = Explorer()
        # explorer.plot_train_vs_validation_loss(train_losses, val_losses, True,
        #                                        self.get_model_unique_name(), "../plots/nn/")

    def fit(self, X_train, y_train, epochs=300):
        """Train the model using the given training data.

        Arguments:
            X_train (numpy ndarray, shape = (samples, 300)):
                Observed portion of timeseries for each data case
            y_train  (numpy ndarray, shape = (samples, 100)):
                Portion of time series to predict for each data case
            step (float):
                step size to use
            epochs (int):
                number of epochs of training
        """
        # input tensor
        x_train_in = torch.tensor(self.__reshape_input_X(X_train)).float()

        # output tensor
        y_train_out = torch.tensor(y_train).float()

        loss_fun = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters["learning_rate"])
        loss_train = np.inf
        train_losses = []

        for epoch in range(epochs):
            # forward pass
            y_train_pred = self.model(x_train_in)
            if epoch == 0:
                print("shape of y_train_pred", y_train_pred.shape)

            # compute the loss
            loss_train = loss_fun(y_train_pred, y_train_out)
            train_losses.append(loss_train)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

    def save_model(self, absolute_filepath):
        torch.save(self.model.state_dict(), absolute_filepath)

    def load_model(self, absolute_filepath):
        # self.model = self.__get_model_class()
        self.model.load_state_dict(torch.load(absolute_filepath))
        self.model.eval()

    def get_model_unique_name(self):
        return "bnn_" + str(self.hyperparameters)

def main():
    DATA_DIR = '../data'

    data = np.load("../data/data_distribute.npz")

    # Training Datat
    X_tr = data['X_tr']
    Y_tr = data['Y_tr']

    # Validation Datat
    X_val = data['X_val']
    Y_val = data['Y_val']

    # For final testing, combine training and validation data
    print(type(X_tr), X_tr.shape, X_val.shape)
    X_tr = np.append(X_tr, X_val, axis=0)
    Y_tr = np.append(Y_tr, Y_val, axis=0)
    print("New dimensions of training data", X_tr.shape, Y_tr.shape)

    # Test data
    # Note: test outputs are not provided in the data set
    X_te = data['X_te']

    # Try computing objective function
    # lr_values = np.logspace(-4, 0, num=5)  # from 1e-4 to 1
    # hl_values = np.array([50, 100, 300, 400, 500])
    hl_values = np.array([500])

    for hl_units in hl_values:
        nn = NN(hl_units, 200, 0.001, 0.9, False, Optimizer.ADAM)
        num_epochs = 51  # the best obtained
        filepath = "../model/nn/" + nn.get_model_unique_name()
        try:
            nn.load_model(filepath)
            print("Loaded learned model from file {}.".format(filepath))
        except Exception as e:
            print("Cound not load model from file {}. Exception {}. Learning again.".format(filepath, e))
            nn.fit_with_validation_data_check(X_tr, Y_tr, X_val, Y_val, epochs=num_epochs)
            nn.save_model(filepath)

        # nn.fit(X_tr, Y_tr, epochs=10)

        Y_tr_pred = nn.predict(X_tr)
        Y_val_pred = nn.predict(X_val)
        print("Loss on training data:", nn.calculate_loss(Y_tr, Y_tr_pred))
        print("Loss on validation data:", nn.calculate_loss(Y_val, Y_val_pred))
        nn.print_parameters()

        # Do test data predictions
        test_pred = nn.predict(X_te)
        np.save("predictions.npy", test_pred)
        print("Test predictions saved.")

        # Plot validation data predictions
        # explorer = Explorer()
        # for i in range(5):
        #     explorer.plot_one_time_series(X_te[i, :], None, test_pred[i, :],
        #                                   True, "nn_test_" + str(num_epochs) + "_" + str(i + 1),
        #                                   directory_path="../plots/nn/")

        # for i in range(5):
        #     explorer.plot_predictions(Y_val[i, :], Y_val_pred[i, :], True,
        #                               filename_suffix=nn.get_model_unique_name()+"val_"+str(i+1),
        #                               directory_path="../plots/nn/")
        # Plot training data predictions
        # for i in range(5):
        #     explorer.plot_predictions(Y_tr[i, :], Y_tr_pred[i, :], False,
        #                               "cnn_" + str(num_epochs) + "_train_" + str(i + 1))



if __name__ == '__main__':
    main()
