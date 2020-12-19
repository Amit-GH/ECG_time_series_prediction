# Time series prediction of ECG data

This repository contains code for predicting ECG time series data. We use multiple models to perform the predictions.
We start with a 2 layer neural network model, and then move to CNN based architecture and RNN based architecture with
LSTM cells. We compare the performances of all the models and see which one behaves the best. The details of the results
can be found in the report `handout/Report.pdf`.

### Running the code

This code was developed on Macbook pro using Conda to manage python environment. To replicate the environment, you
can use the `spec-file.txt`. See [building identical conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments) for help.

After downloading the code, you can directly run the `main()` methods of `nn.py`, `cnn.py` and `rnn.py` to see their 
prediction performance against the dataset. We use L2 regularization loss for evaluation.
