import os
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from utility.Explorer import Explorer
from enum import Enum

# to remove MacOS specific warnings for MKL
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.random.seed(1)
torch.manual_seed(1)


class Optimizer(Enum):
    ADAM = 1
    SGD = 2
    RMSPROP = 3


class RNN:
    """A neural network model for timeseries forecasting

    Arguments:
        None (add what you need)
    """
    model = None
    hidden_state = None
    num_layers = None
    model_unique_name = None
    linear_units = None
    opt = None

    # leaned parameters of the model
    learned_paramters = {
        "min_val_loss": None,
        "min_val_loss_epoch": None
    }

    def __init__(self, input_size, hidden_state, num_layers, opt=Optimizer.ADAM):
        self.hidden_state = hidden_state
        self.num_layers = num_layers
        self.linear_units = 100
        self.opt = opt
        self.learned_paramters = {
            "min_val_loss": np.inf,
            "min_val_loss_epoch": -1
        }

        self.model = torch.nn.Sequential(
            nn.LSTM(input_size, hidden_state, num_layers, batch_first=True),
            ExtractLastLayerOfRNN(),
            nn.Linear(hidden_state, self.linear_units),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.linear_units, 100)
        )
        print(self.model)

    def objective(self, X, y):
        """Compute objective function.

        Arguments:
            X (numpy ndarray, shape = (samples, 300)):
                Observed portion of timeseries for each data case.
            y  (numpy ndarray, shape = (samples,100)):
                Portion of time series to predict for each data case

        Returns:
            float: Mean squared error objective value. Mean is taken
            over all dimensions of all data cases.
        """
        y_pred = self.predict(X)
        return self.calculate_loss(y, y_pred)

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

    def fit(self, X_train, y_train, X_val, y_val, step=1e-3, epochs=400):
        """Train the model using the given training data.

        Arguments:
            X_train (numpy ndarray, shape = (samples, 300)):
                Observed portion of timeseries for each data case
            y_train  (numpy ndarray, shape = (samples,100)):
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
        optimizer = self.get_optimizer()

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
            if epoch % 15 == 0:
                print("After epoch {}, train loss is {}, val loss is {}.".format(epoch, loss_train, loss_val))

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
        explorer.plot_train_vs_validation_loss(train_losses, val_losses, True,
                                               self.get_model_unique_name(), "../plots/rnn/")

    def get_model_unique_name(self):
        """
        This returns model name useful for saving plots appended by model name.
        Returns: A string that identifies the model by its hyperparameters
        """
        if self.model_unique_name is None:
            self.model_unique_name = "rnn_{}_{}_{}_{}".format(self.opt, self.hidden_state, self.num_layers, self.linear_units)
        return self.model_unique_name

    def get_optimizer(self):
        if self.opt == Optimizer.ADAM:
            return torch.optim.Adam(self.model.parameters())
        elif self.opt == Optimizer.SGD:
            return torch.optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9)
        elif self.opt == Optimizer.RMSPROP:
            return torch.optim.RMSprop(self.model.parameters())

    def load_model(self, filepath):
        self.model = torch.load(filepath)
        self.model.eval()
        self.model_unique_name = filepath.split(sep="/")[-1]

    def save_model(self, filepath):
        torch.save(self.model, filepath)


class ExtractLastLayerOfRNN(nn.Module, ABC):
    def forward(self, lstm_output):
        output, (h_n, c_n) = lstm_output
        return output.view(output.size(0), -1)


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
    num_layers_list = [1, 2]
    hidden_states_list = [200, 300, 400, 500]
    for num_layers in num_layers_list:
        for hidden_states in hidden_states_list:
            rnn = RNN(input_size=300, hidden_state=hidden_states, num_layers=num_layers, opt=Optimizer.ADAM)
            num_epochs = 200
            filepath = "../model/rnn/" + rnn.get_model_unique_name()
            save_model = True
            try:
                rnn.load_model(filepath)
                print("Loaded learned model from file {}.".format(filepath))
            except Exception as e:
                print("Count not load model from file {}. Exception {}. Learning again.".format(filepath, e))
                rnn.fit(X_tr, Y_tr, X_val, Y_val, epochs=num_epochs)
                if save_model:
                    rnn.save_model(filepath)

            # Do prediction on train, validation and test data
            Y_tr_pred = rnn.predict(X_tr)
            Y_val_pred = rnn.predict(X_val)
            Y_te_pred = rnn.predict(X_te)
            print("Loss on training data", rnn.calculate_loss(Y_tr, Y_tr_pred))
            print("Loss on validation data", rnn.calculate_loss(Y_val, Y_val_pred))

            # Plot validation data predictions
            explorer = Explorer()
            for i in range(2):
                explorer.plot_predictions(
                    Y_val[i, :],
                    Y_val_pred[i, :],
                    save_fig=True,
                    filename_suffix=(rnn.get_model_unique_name() + "_val_" + str(i + 1)),
                    directory_path="../plots/rnn/"
                )


if __name__ == '__main__':
    main()
