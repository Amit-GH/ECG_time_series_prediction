import numpy as np
import matplotlib.pyplot as plt

class Explorer:
    def explore_data(self, X, y, title="Input data"):
        print(title)
        print("X shape", X.shape)
        print("y shape", y.shape)
        print("X element type", type(X[0][0]))
        print("y element type", type(y[0][0]))

    def plot_predictions(self, y, y_pred, save_fig=False, filename_suffix="", directory_path="../plots/"):
        x = np.arange(start=1, stop=len(y) + 1)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y, color='green', label="Actual output")
        ax.plot(x, y_pred, color='red', label="Predicted output")
        ax.legend()
        ax.set(
            xlabel=r'$Time Points$',
            ylabel=r'$Voltage$',
            title="ECG prediction vs actual values for a time range",
        )
        if save_fig:
            plt.savefig(directory_path + "prediction_plot_" + filename_suffix + ".jpg", format='jpg', dpi=300)
        else:
            plt.show()

    def plot_one_time_series(self, x, y, y_pred, save_fig=False, filename_suffix="",
                             directory_path="../plots/"):
        x_axis_train = np.arange(start=1, stop=len(x) + 1)
        x_axis_test = np.arange(start=len(x), stop=len(x) + len(y_pred))
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        y_axis_train = x
        y_axis_test = y
        y_axis_test_pred = y_pred
        ax.plot(x_axis_train, y_axis_train, color='green', label="Given training series")
        if y_axis_test is not None:
            ax.plot(x_axis_test, y_axis_test, color='red', label="Test series")
            title = "ECG predictions for a time range"
        else:
            title = "ECG prediction vs actual values for a time range"
        ax.plot(x_axis_test, y_axis_test_pred, color='blue', label="Predicted test series")
        ax.legend()
        ax.set(
            xlabel=r'$Time Points$',
            ylabel=r'$Voltage$',
            title=title,
        )
        if save_fig:
            plt.savefig(directory_path + "time_series_pred_" + filename_suffix + ".jpg", format='jpg', dpi=300)
        else:
            plt.show()

    def plot_train_vs_validation_loss(self, train_losses, val_losses, save_fig=False, filename_suffix="",
                                      directory_path="../plots/"):
        """
        Args:
            train_losses: numpy list of training losses
            val_losses: numpy list of validation losses

        Returns:
            Plots the training vs validation loss curve
        """

        assert len(train_losses) == len(val_losses)
        epochs = len(train_losses)
        x_axis = np.arange(1, epochs+1)
        y_axis_train_loss = train_losses
        y_axis_val_loss = val_losses
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_axis, y_axis_train_loss, color='green', label="Training loss")
        ax.plot(x_axis, y_axis_val_loss, color='red', label="Validation loss")
        ax.legend()
        ax.set(
            xlabel="Number of Epochs",
            ylabel="Loss",
            title="Training and Validation set loss with epochs"
        )
        if save_fig:
            plt.savefig(directory_path + "train_val_loss_" + filename_suffix + ".jpg", format='jpg', dpi=300)
        else:
            plt.show()
