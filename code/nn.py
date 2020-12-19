import numpy as np
import torch
import os
# import matplotlib.pyplot as plt

# to remove MacOS specific warnings for MKL
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.random.seed(1)
torch.manual_seed(1)

'''
Some helpful links:
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#nn-module
'''


class NN:
    """A neural network model for timeseries forecasting

    Arguments:
        h1_size (int): Size of the hidden layer
    """
    h1_size = None
    w1, b1, w2, b2 = None, None, None, None
    model = None

    def __init__(self, h1_size=100):
        self.h1_size = h1_size

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
        return np.mean(np.square(y - y_pred))

    def predict(self, X):
        """Forecast time series values.

        Arguments:
            X (numpy ndarray, shape = (samples, 300)):
                Observed portion of timeseries for each data case.

        Returns:
            y  (numpy ndarray, shape = (samples, 100)):
                Predicted portion of time series for each data case.
        """
        x_in = torch.tensor(X).float()
        y_out = self.model(x_in)
        y = y_out.detach().numpy()
        # print("predict y shape", y.shape)
        # print("predict y type", type(y))
        return y

    def fit(self, X, y, step=1e-3, epochs=400):
        """Train the model using the given training data.

        Arguments:
            X (numpy ndarray, shape = (samples, 300)):
                Observed portion of timeseries for each data case
            y  (numpy ndarray, shape = (samples,100)):
                Portion of time series to predict for each data case
            step (float):
                step size to use
            epochs (int):
                number of epochs of training
        """
        # RMSProp optimizer parameters
        weight_decay = 1e-4
        momentum = 0.9

        N = X.shape[0]
        num_features = X.shape[1]
        num_outputs = y.shape[1]

        # input tensor
        x_in = torch.tensor(X).float()
        # print(x_in.shape)
        # print(type(x_in))
        # print(x_in.dtype)

        # output tensor
        y_out = torch.tensor(y).float()

        # initialize the model
        self.initialize_model(num_features, num_outputs)

        loss_fun = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=step, weight_decay=weight_decay, momentum=momentum)
        loss = np.inf

        for i in range(epochs):
            # forward pass
            y_pred = self.model(x_in)

            # compute the loss
            loss = loss_fun(y_pred, y_out)
            if i % 50 == 0:
                print("loss after epoch", i, loss)

            # zero the gradients (as per recommendation)
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update the model parameters
            optimizer.step()

        print("Final loss of model", loss)

        # Extract the model parameters from the model. The weights are saved in row major form in the model
        # so we need to take the transpose to meet our expectations.
        all_param = list(self.model.parameters())
        assert len(all_param) == 4, "Number of model parameters should be 4."
        self.w1 = all_param[0].detach().numpy().transpose()
        self.b1 = all_param[1].detach().numpy()
        self.w2 = all_param[2].detach().numpy().transpose()
        self.b2 = all_param[3].detach().numpy()

    def get_params(self):
        """Get the model parameters.

        Returns:
            a list [w1,b1,w2,b2] containing the following 
            parameter values represented as numpy arrays 
            (see handout for definitions of each parameter). 
        
            w1 (numpy ndarray, shape = (300, h1_size))
            b1 (numpy ndarray, shape = (h1_size,))
            w2 (numpy ndarray, shape = (h1_size, 100))
            b2 (numpy ndarray, shape = (100,))
        """
        return [self.w1, self.b1, self.w2, self.b2]

    def set_params(self, params):
        """Set the model parameters.

        Arguments:
            params is a list [w1,b1,w2,b2] containing the following 
            parameter values represented as numpy arrays (see handout 
            for definitions of each parameter).
        
            w1 (numpy ndarray, shape = (300, h1_size))
            b1 (numpy ndarray, shape = (h1_size,))
            w2 (numpy ndarray, shape = (h1_size, 100))
            b2 (numpy ndarray, shape = (100,))
        """
        assert len(params) == 4
        self.w1, self.b1, self.w2, self.b2 = params[0], params[1], params[2], params[3]
        state_dict = {
            '0.weight': torch.tensor(np.transpose(self.w1)),
            '0.bias': torch.tensor(self.b1),
            '2.weight': torch.tensor(np.transpose(self.w2)),
            '2.bias': torch.tensor(self.b2)
        }
        num_features = self.w1.shape[0]
        num_outputs = self.w2.shape[1]
        self.initialize_model(num_features, num_outputs)
        self.model.load_state_dict(state_dict)
        print("Model parameters have been manually set.")

    def initialize_model(self, num_features, num_outputs):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_features, self.h1_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h1_size, num_outputs)
        )

    def see_model_parameters(self):
        print("Inside see model parameters.")
        state_dict = self.model.state_dict()
        print(state_dict.keys())
        w1_key_name = list(state_dict.keys())[0]
        w1 = state_dict[w1_key_name]
        print(w1_key_name, "shape", w1.shape)
        print(w1_key_name, "type", type(w1))

    def print_model(self):
        print(self.model)


# class Explorer:
#     def explore_data(self, X, y, title="Input data"):
#         print(title)
#         print("X shape", X.shape)
#         print("y shape", y.shape)
#         print("X element type", type(X[0][0]))
#         print("y element type", type(y[0][0]))
#
#     def plot_predictions(self, y, y_pred, save_fig=False, filename_suffix=""):
#         x = np.arange(start=1, stop=len(y) + 1)
#         fig = plt.figure(figsize=(6, 6))
#         ax = fig.add_subplot(1, 1, 1)
#         ax.plot(x, y, color='green', label="Actual output")
#         ax.plot(x, y_pred, color='red', label="Predicted output")
#         ax.legend()
#         ax.set(
#             xlabel='Time Points',
#             ylabel='Voltage',
#             title="ECG predicted vs actual values for validation set " + filename_suffix,
#         )
#         if save_fig:
#             plt.savefig("../plots/prediction_plot_" + filename_suffix + ".jpg", format='jpg', dpi=300)
#         else:
#             plt.show()
#
#     def plot_one_time_series(self, x, y, y_pred, save_fig=False, filename_suffix=""):
#         x_axis_train = np.arange(start=1, stop=len(x) + 1)
#         x_axis_test = np.arange(start=len(x), stop=len(x) + len(y))
#         fig = plt.figure(figsize=(6, 6))
#         ax = fig.add_subplot(1, 1, 1)
#         y_axis_train = x
#         y_axis_test = y
#         y_axis_test_pred = y_pred
#         ax.plot(x_axis_train, y_axis_train, color='green', label="Given training series")
#         ax.plot(x_axis_test, y_axis_test, color='red', label="Test series")
#         ax.plot(x_axis_test, y_axis_test_pred, color='blue', label="Predicted test series")
#         ax.legend()
#         ax.set(
#             xlabel=r'$Time Points$',
#             ylabel=r'$Voltage$',
#             title="ECG prediction vs actual values for a time range",
#         )
#         if save_fig:
#             plt.savefig("../plots/time_series_pred_" + filename_suffix + ".jpg", format='jpg', dpi=300)
#         else:
#             plt.show()
#
#     def plot_train_vs_validation_loss(self, train_losses, val_losses, save_fig=False, filename_suffix=""):
#         """
#         Args:
#             train_losses: numpy list of training losses
#             val_losses: numpy list of validation losses
#
#         Returns:
#             Plots the training vs validation loss curve
#         """
#
#         assert len(train_losses) == len(val_losses)
#         epochs = len(train_losses)
#         x_axis = np.arange(1, epochs+1)
#         y_axis_train_loss = train_losses
#         y_axis_val_loss = val_losses
#         fig = plt.figure(figsize=(6, 6))
#         ax = fig.add_subplot(1, 1, 1)
#         ax.plot(x_axis, y_axis_train_loss, color='green', label="Training loss")
#         ax.plot(x_axis, y_axis_val_loss, color='red', label="Validation loss")
#         ax.legend()
#         ax.set(
#             xlabel="Number of Epochs",
#             ylabel="Loss",
#             title="Training and Validation set loss with epochs"
#         )
#         if save_fig:
#             plt.savefig("../plots/train_val_loss_" + filename_suffix + ".jpg", format='jpg', dpi=300)
#         else:
#             plt.show()


def main():
    DATA_DIR = '../data'

    data = np.load("../data/data_distribute.npz")
    # explorer = Explorer()

    # Training Data
    X_tr = data['X_tr']
    Y_tr = data['Y_tr']

    # explorer.explore_data(X_tr, Y_tr, title='Training data')

    # Validation Data
    X_val = data['X_val']
    Y_val = data['Y_val']

    # explorer.explore_data(X_val, Y_val, title='Validation data')

    # Test data
    # Note: test outputs are not provided in the data set
    X_te = data['X_te']

    # Try setting params
    nn = NN(h1_size=100)
    nn.set_params([np.random.randn(300, 100) / 10,
                   np.random.randn(100) / 10,
                   np.random.randn(100, 100) / 10,
                   np.random.randn(100) / 10])
    print("b2 set (5 values)", nn.get_params()[3][:5])

    # Try computing objective function
    print("Obj:", nn.objective(X_tr, Y_tr))

    # Try predicting
    Y_tr_hat = nn.predict(X_tr)
    print("Y_tr_hat shape", Y_tr_hat.shape)

    # Try fitting for question 2.b.
    nn = NN(h1_size=100)
    try:
        saved_param = []
        with open("param_ques_2b.npy", "rb") as f:
            for i in range(4):
                saved_param.append(np.load(f))
        print("Using saved parameters.")
        nn.set_params(saved_param)
    except Exception as e:
        print("Encountered exception while loading parameters for ques 2.b.", e)
        print("Training the model to find parameters.")
        nn.fit(X_tr, Y_tr, epochs=400)
        nn.see_model_parameters()
        learned_param = nn.get_params()
        with open("param_ques_2b.npy", "wb") as f:
            for param in learned_param:
                np.save(f, param)

    nn.print_model()

    print("b2 set (5 values)", nn.get_params()[3][:5])
    loss_tr = nn.objective(X_tr, Y_tr)
    print("2.b. training set loss", loss_tr)
    loss_val = nn.objective(X_val, Y_val)
    print("2.b. validation set loss", loss_val)

    # Plot graphs for question 2.c.
    for i in range(5):
        x_val = X_val[i, :]
        y_val = Y_val[i, :]
        y_val_pred = nn.predict(x_val)
        if i == 0:
            print("Shape of data one validation data:", x_val.shape, y_val.shape, y_val_pred.shape)
        save_fig = True
        # explorer.plot_predictions(y_val, y_val_pred, save_fig=save_fig, filename_suffix=str(i + 1))

    # Plot a complete time-series having both input and output parts.
    for i in range(5):
        x_val = X_val[i, :]
        y_val = Y_val[i, :]
        y_val_pred = nn.predict(x_val)
        # explorer.plot_one_time_series(x_val, y_val, y_val_pred, save_fig=False, filename_suffix=str(i + 1))

    # Try getting parameters
    out = nn.get_params()
    # print("Learned parameters: ", out)

    # Try saving predictions
    # pred = nn.predict(X_te)
    # print("Obj:", nn.objective(X_tr, Y_tr))
    # np.save("predictions.npy", pred)


if __name__ == '__main__':
    main()
