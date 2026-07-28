import numpy as np
from models.ARIMA import Model
from dataset import data_provider

def configure_model_params(args):
    if args.task_name == 'anomaly_detection':
        args.seq_len = 100
        args.pred_len = 0
        args.d_model = 128
        args.d_ff = 128
        args.e_layers = 3
        args.enc_in = 55
        args.c_out = 55
        args.anomaly_ratio = 1
        args.batch_size = 128
        args.train_epochs = 10
        return
    if args.model in ['TimesNet', 'LightTS', 'DLinear', 'LSTM', 'CNN', 'Linear']:
        args.dff = 512
        args.d_model = 256
        args.e_layers = 2
        args.d_layers = 1
        args.factor = 3
        args.enc_in = 1  # univariate
        args.dec_in = 1
        args.c_out = 1
        args.top_k = 5
    if args.model == 'Informer':
        args.e_layers = 3
        args.batch_size = 16
        args.d_model = 128
        args.d_ff = 256
        args.top_k = 3
        args.learning_rate = 0.001
        args.train_epochs = 100
        args.patience = 10
    if args.model == 'Nonstationary_Transformer' or 'Autoformer':
        args.dff = 512
        args.d_model = 256
        args.e_layers = 2
        args.d_layers = 1
        args.factor = 3
        args.enc_in = 1  # univariate
        args.dec_in = 1
        args.c_out = 1
        args.train_epochs = 3
    if args.model == 'PatchTST':
        args.dff = 512
        args.d_model = 256
        args.e_layers = 2
        args.d_layers = 1
        args.factor = 3
        args.enc_in = 1  # univariate
        args.dec_in = 1
        args.c_out = 1
        args.batch_size = 16
    if args.model == 'TimeMixer':
        args.dff = 32
        args.d_model = 16
        args.e_layers = 3
        args.d_layers = 1
        args.factor = 3
        args.enc_in = 1  # input dimension
        args.dec_in = 1
        args.c_out = 1
        args.batch_size = 32
        args.learning_rate = 0.01
        args.train_epochs = 20
        args.patience = 10
        args.down_sampling_layers = 3
        args.down_sampling_method = 'avg'
        args.down_sampling_window = 2
        args.label_len = 0
    if args.model == 'iTransformer':
        args.d_model = 512
        args.d_ff = 512
        args.batch_size = 16
        args.learning_rate = 0.0005
        args.enc_in = 1  # input dimension
        args.dec_in = 1
        args.c_out = 1
        args.e_layers = 3
        args.d_layers = 1
        args.factor = 3


def print_experiment_results(score_keys, proportions, results, num_iterations):
    """
    Print results for every experiment iteration.
    :param score_keys: List of score metric names.
    :param proportions: Data proportions to evaluate.
    :param results: Dictionary containing experiment results.
    :param num_iterations: Total number of experiment iterations.
    """
    print("\nExperiment results")

    # Set up the table header.
    header = ["Score Key"] + proportions

    # Use a common column width for alignment.
    column_width = max(len(str(item)) for item in header)

    # Print the RMSE for every metric and data proportion in each iteration.
    for i in range(num_iterations):
        print("\n" + f"Iteration {i + 1}")
        print("\t".join([f"{item:<{column_width}}" for item in header]))  # Print header.
        for score_key in score_keys:
            row = [score_key]
            for proportion in proportions:
                rmse_list = results[score_key][proportion]
                value = f"{rmse_list[i]:.3f}" if i < len(rmse_list) else "N/A"
                row.append(value)
            print("\t".join([f"{item:<{column_width}}" for item in row]))

    # Print mean results.
    print("\n" + "Mean results")
    print("\t".join([f"{item:<{column_width}}" for item in header]))  # Print header.
    for score_key in score_keys:
        row = [score_key]
        for proportion in proportions:
            rmse_list = results[score_key][proportion]
            if rmse_list:
                average_rmse = sum(rmse_list) / len(rmse_list)
                row.append(f"{average_rmse:.3f}")
            else:
                row.append("N/A")
        print("\t".join([f"{item:<{column_width}}" for item in row]))

    # Print minimum results.
    print("\nMinimum results")
    print("\t".join([f"{item:<{column_width}}" for item in header]))  # Print header.
    for score_key in score_keys:
        row = [score_key]
        for proportion in proportions:
            rmse_list = results[score_key][proportion]
            if rmse_list:
                min_rmse = min(rmse_list)
                row.append(f"{min_rmse:.3f}")
            else:
                row.append("N/A")
        print("\t".join([f"{item:<{column_width}}" for item in row]))



def arima_training_and_testing(args, setting):
    """
    Function to train and test ARIMA model using the entire training set.
    :param args: Configuration arguments
    :param setting: Experiment setting string
    :param rmse_list: List to store RMSE values
    """
    print(f">>>>>>> Start ARIMA model training and testing: {setting} >>>>>>>>>>>>>>>>>")
    # args.proportion = 0.1
    args.temperature = 2.0
    # Load the training data through data_provider.
    data_set, data_loader = data_provider(args, flag="train")

    # Create the ARIMA model instance.
    arima_model = Model(args)

    listrmse = []
    # Train and forecast each series in the training dataset.
    for i, data_point in enumerate(data_set):
        # data_point is a four-tuple; only x_enc and y_true are used here.
        x_enc, y_true, _, _ = data_point
        y_true = y_true[len(y_true) // 2:]

        # Fit ARIMA on x_enc and forecast the target horizon using the first feature.
        forecast = arima_model.arima_forecast(x_enc[:, 0])

        # Compute RMSE between the forecast and the target values.
        rmse = np.sqrt(((forecast - y_true[:, 0]) ** 2).mean())
        listrmse.append(rmse)

    rmse_avg = np.mean(listrmse)

    return rmse_avg


