import os
from abc import ABC

import numpy as np
import torch
from torch.nn import Conv1d, Linear, Sequential, ReLU, MaxPool1d, Module
from utility.Explorer import Explorer

# to remove MacOS specific warnings for MKL
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.random.seed(1)
torch.manual_seed(1)


class CNN:
    """A neural network model for timeseries forecasting

    Arguments:
        None (add what you need)
    """
    model = None
    model_unique_name = None

    # leaned parameters of the model
    learned_paramters = {
        "min_val_loss": None,
        "min_val_loss_epoch": None
    }

    def __init__(self, oc1, ks1, sd1, mp1, oc2, ks2, sd2, mp2, lu):
        self.model = self.__get_model_class(oc1, ks1, sd1, mp1, oc2, ks2, sd2, mp2, lu)
        self.learned_paramters = {
            "min_val_loss": np.inf,
            "min_val_loss_epoch": -1
        }
        print(self.model)

    def __get_model_class(self, oc1, ks1, sd1, mp1, oc2, ks2, sd2, mp2, lu):
        # oc1 = 4
        # ks1 = 3
        # sd1 = 1
        # mp1 = 2
        # oc2 = 4
        # ks2 = 3
        # sd2 = 1
        # mp2 = 2
        self.model_unique_name = "cnn_2blocks_{}_{}_{}_{}_and_{}_{}_{}_{}".format(
            oc1, ks1, sd1, mp1, oc2, ks2, sd2, mp2
        )
        return Sequential(
            Conv1d(in_channels=1, out_channels=oc1, kernel_size=ks1, stride=sd1),
            ReLU(),
            MaxPool1d(mp1, stride=1),
            Conv1d(in_channels=oc1, out_channels=oc2, kernel_size=ks2, stride=sd2),
            ReLU(),
            MaxPool1d(mp2, stride=1),
            Conv1DToLinearConverter(),
            Linear(lu, 100),  # input size needs to be adjusted based on other parameters
            # ReLU(inplace=True),
            # Linear(300, 100)
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
        num_samples = X.shape[0]
        num_features = X.shape[1]
        num_channels = 1
        return np.reshape(X, (num_samples, num_channels, num_features))

    def print_parameters(self):
        parameters = list(self.model.parameters())
        print("num parameters ", len(parameters))
        # print(self.model.state_dict())
        # print("parameters list:", parameters)

    def fit(self, X_train, y_train, X_val, y_val, epochs=400):
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
        optimizer = torch.optim.Adam(self.model.parameters())
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
                print("Loss after epoch ", epoch, loss_train)

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
        explorer = Explorer()
        explorer.plot_train_vs_validation_loss(train_losses[3:], val_losses[3:], True,
                                               self.get_model_unique_name(), "../plots/cnn/")

    def save_model(self, absolute_filepath):
        torch.save(self.model.state_dict(), absolute_filepath)

    def load_model(self, absolute_filepath):
        # self.model = self.__get_model_class()
        self.model.load_state_dict(torch.load(absolute_filepath))
        self.model.eval()

    def get_model_unique_name(self):
        return self.model_unique_name


class Conv1DToLinearConverter(Module, ABC):
    def forward(self, X):
        return X.view(X.size(0), -1)


def main():
    DATA_DIR = '../data'

    data = np.load("../data/data_distribute.npz")

    # Training Datat
    X_tr = data['X_tr']
    Y_tr = data['Y_tr']

    # Validation Datat
    X_val = data['X_val']
    Y_val = data['Y_val']

    # Test data
    # Note: test outputs are not provided in the data set
    X_te = data['X_te']

    # Try computing objective function
    hyperparam = [
        [4, 3, 1, 2, 4, 3, 1, 2, 1176],
        [10, 3, 1, 2, 10, 3, 1, 2, 2940],
        [20, 3, 1, 2, 20, 3, 1, 2, 5880],
        [10, 5, 1, 2, 10, 5, 1, 2, 2900],
        [10, 10, 1, 2, 10, 10, 1, 2, 2800],
        [10, 20, 1, 2, 10, 20, 1, 2, 2600],
        [10, 10, 1, 3, 10, 10, 1, 3, 2780],
        [10, 10, 1, 5, 10, 10, 1, 5, 2740],
        [10, 10, 1, 8, 10, 10, 1, 8, 2680],
    ]
    for hp in hyperparam:
        oc1, ks1, sd1, mp1, oc2, ks2, sd2, mp2, lu = hp
        nn = CNN(oc1, ks1, sd1, mp1, oc2, ks2, sd2, mp2, lu)
        num_epochs = 100
        filepath = "../model/cnn/" + nn.get_model_unique_name()
        try:
            nn.load_model(filepath)
            print("Loaded learned model from file {}.".format(filepath))
        except Exception as e:
            print("Count not load model from file {}. Exception {}. Learning again.".format(filepath, e))
            nn.fit(X_tr, Y_tr, X_val, Y_val, epochs=num_epochs)
            nn.save_model(filepath)

        Y_tr_pred = nn.predict(X_tr)
        Y_val_pred = nn.predict(X_val)
        print("Loss on training data:", nn.calculate_loss(Y_tr, Y_tr_pred))
        print("Loss on validation data:", nn.calculate_loss(Y_val, Y_val_pred))
        nn.print_parameters()

        explorer = Explorer()
        # Plot validation data predictions
        for i in range(2):
            explorer.plot_predictions(Y_val[i, :], Y_val_pred[i, :], True,
                                      filename_suffix=nn.get_model_unique_name() + "val_" + str(i + 1),
                                      directory_path="../plots/cnn/")

        # for i in range(5):
        #     explorer.plot_one_time_series(X_val[i, :], Y_val[i, :], Y_val_pred[i, :],
        #                                   True, "cnn_" + str(num_epochs) + "_" + str(i+1))
        #
        # # Plot training data predictions
        # for i in range(5):
        #     explorer.plot_predictions(Y_tr[i, :], Y_tr_pred[i, :], False,
        #                               "cnn_" + str(num_epochs) + "_train_" + str(i+1))


if __name__ == '__main__':
    main()
