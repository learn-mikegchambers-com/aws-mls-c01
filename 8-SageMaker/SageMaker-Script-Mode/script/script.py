import argparse
import os
# from sagemaker_training import environment
from sklearn.linear_model import LinearRegression
import pickle

def parse_args():
    """
    Parse arguments.
    """
    #env = environment.Environment()

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    # We don't use these but I left them in as a useful template for future development
    parser.add_argument("--copy_X",        type=bool, default=True)
    parser.add_argument("--fit_intercept", type=bool, default=True)
    parser.add_argument("--normalize",     type=bool, default=False)
    
    # data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_known_args()

def load_dataset(path):
    """
    Load entire dataset.
    """
    # Find all files with a pickle ext but we only load the first one in this sample:
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith("pickle")]

    if len(files) == 0:
        raise ValueError("Invalid # of files in dir: {}".format(path))
    
    [X, y] = pickle.load(open(files[0], 'rb'))
    
    return X, y

def start(args):
    """
    Train a Linear Regression
    """
    print("Training mode")

    try:
        X_train, y_train = load_dataset(args.train)
        # X_test, y_test = load_dataset(args.test)

        hyperparameters = {
            "copy_X": args.copy_X,
            "fit_intercept": args.fit_intercept,
            "normalize": args.normalize,
        }
        
        print("Training...")
        model = LinearRegression()
        model.set_params(**hyperparameters)
                
        model.fit(X_train, y_train)

        pickle.dump(model, open(os.path.join(args.model_dir, "model.pickle"), 'wb'))
       

    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, "failure"), "w") as s:
            s.write("Exception during training: " + str(e) + "\\n" + trc)

        # Printing this causes the exception to be in the training job logs, as well.
        print("Exception during training: " + str(e) + "\\n" + trc, file=sys.stderr)

        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)
        
def model_fn(model_dir):
    """
    Load the model for inference
    """
    loaded_model = pickle.load(open(model_dir + "/model.pickle", 'rb'))
    return loaded_model


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    return model.predict(input_data)


if __name__ == "__main__":
    
    args, _ = parse_args()

    start(args)