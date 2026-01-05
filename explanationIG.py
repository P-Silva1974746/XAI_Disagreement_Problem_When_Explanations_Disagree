import openml
import logging
import sys
import joblib
import numpy as np
import pandas as pd
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from sklearn.model_selection import train_test_split
from collections import defaultdict

# necessary to be here since the when loading the models joblid tries to import  __main__.to_float32_fn it wouldn't find it here
# this happens because the MLP model pipeline depdends on this function
def to_float32_fn(X):
    return X.astype(np.float32)

#same reason as above
class MLP_IG(nn.Module):
    def __init__(self, output_dim, hidden_dim=16):
        super().__init__()

        self.fc1 = nn.LazyLinear(hidden_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.out = nn.Linear(16, output_dim)

        # # better initialization for IG smoothness
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, nn.LazyLinear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.out(x)  # no softmax
        return logits
    

def load_models(model_dir, dataset_id):
    models = {
        "lr": joblib.load(f"{model_dir}/lr_{dataset_id}.joblib"),
        "rf": joblib.load(f"{model_dir}/rf_{dataset_id}.joblib"),
        "mlp": joblib.load(f"{model_dir}/mlp_{dataset_id}.joblib")
    }
    return models

def explain_with_ig(model_pipeline, X_explain, target_class=1):
    """
    Compute Integrated Gradients for a PyTorch model wrapped in a sklearn or skorch pipeline.

    Parameters:
    -----------
    model_pipeline : sklearn.pipeline.Pipeline or PyTorch nn.Module
        If Pipeline, the last step must be a PyTorch model (skorch NeuralNetClassifier or nn.Module)
    X_explain : pd.DataFrame or np.ndarray
        The raw input data to explain (untransformed)
    target_class : int
        The target class index for which to compute attributions

    Returns:
    --------
    np.ndarray
        Integrated Gradients attributions (shape: n_samples x n_features)
    """

    # If it is a Pipeline, separate preprocessing and model
    if hasattr(model_pipeline, 'named_steps'):
        preprocess = model_pipeline.named_steps['preprocess']
        X_trans = preprocess.transform(X_explain)
        # convert sparse to dense if necessary
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        # get the model (unwrap skorch if necessary)
        model = model_pipeline.named_steps['clf']
    else:
        # assume X_explain is already transformed
        X_trans = X_explain
        model = model_pipeline

    # unwrap skorch model to get raw nn.Module
    if hasattr(model, "module_"):
        model = model.module_

    model.eval()
    ig = IntegratedGradients(model)

    attributions = []
    for x in X_trans:
        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        attr = ig.attribute(x_tensor, target=target_class)
        attributions.append(attr.detach().numpy())

    return np.vstack(attributions)


def aggregate_attributions(attributions, feature_groups):
    """
    attributions: np.ndarray of shape (n_samples, n_encoded_features)
    feature_groups: dict {original_feature: [indices]}
    """
    agg = np.zeros((attributions.shape[0], len(feature_groups)))
    feature_names = list(feature_groups.keys())

    for i, feature in enumerate(feature_names):
        idxs = feature_groups[feature]
        agg[:, i] = np.abs(attributions[:, idxs]).sum(axis=1)

    return agg, feature_names



def build_feature_groups(feature_names, colname, categorical_mask):
    """
    Maps original features to lists of encoded feature indices.
    Correctly handles categorical vs numerical features and avoids substring collisions.
    """
    groups = defaultdict(list)

    # Build lookup sets
    categorical_cols = {
        colname[i] for i, is_cat in enumerate(categorical_mask) if is_cat
    }
    numerical_cols = {
        colname[i] for i, is_cat in enumerate(categorical_mask) if not is_cat
    }

    for idx, name in enumerate(feature_names):
        # Remove transformer prefix (e.g., "num__", "cat__")
        clean = name.split("__", 1)[-1]

        # Case 1: numerical feature → exact match
        if clean in numerical_cols:
            original = clean

        # Case 2: categorical feature → prefix before first "_" must match a categorical column
        else:
            base = clean.rsplit("_", 1)[0]
            if base in categorical_cols:
                original = base
            else:
                raise ValueError(f"Could not map encoded feature '{name}' to original column")

        groups[original].append(idx)

    return dict(groups)



if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("explanations_IG.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)


    datasets= [1504, 1590, 1510, 37, 31, 59, 45547, 44, 1558, 40981]

    for id in datasets:

        logger.info(f"=== Starting dataset {id} ===")

        try:
            dataset = openml.datasets.get_dataset(dataset_id=id, download_data=True, download_qualities=True, download_features_meta_data=True)
            X, y, categorical_mask, colname=dataset.get_data(target=dataset.default_target_attribute , dataset_format="dataframe")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
            col_names= X_train.columns.tolist()

            logger.info(f"Dataset {dataset.name} (id: {id})")
            models = load_models(model_dir="models", dataset_id=id)

            # Explanation sets
            X_explain = X_test.iloc[:50]

            ig_mlp_vals = explain_with_ig(models["mlp"], X_explain, target_class=1)
            logger.info(f"\tCalculated explanations")


            encoded_feature_names = models['lr'].named_steps['preprocess'].get_feature_names_out()
            feature_groups = build_feature_groups(encoded_feature_names, colname=colname, categorical_mask=categorical_mask)

            agg_ig_vals_mlp, agg_feature_names = aggregate_attributions(
                ig_mlp_vals,
                feature_groups
            )

            agg_mean_ig_vals_mlp=[]

            for i in range(agg_ig_vals_mlp.shape[1]):
                agg_mean_ig_vals_mlp.append(agg_ig_vals_mlp[:,i].mean())

            res = pd.DataFrame(data=agg_ig_vals_mlp, columns=colname)

            res.to_csv(f"results/IG_{id}.csv")
            logger.info(f"\nSaved results\n")
            logger.info("\n\n\n#------------------------------------------------------------------------------#\n\n\n")


        except Exception as e:
            logger.error(
                f"Error while processing dataset {id}",
                exc_info=True
            )
        
        finally:
                # Always clean memory, even if it failed halfway
                #because (will prevent the models and datasets of persisting in memory which could lead to memory overusage)
            for var in [
                "dataset", "X", "y", "X_train", "X_test", "y_train", "y_test",
                "models", "X_explain", "ig_mlp_vals",
                "encoded_feature_names", "feature_groups", "agg_ig_vals_mlp",
                "agg_feature_names", "agg_mean_ig_vals_mlp", "res",
            ]:
                if var in locals():
                    del locals()[var]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

            logger.info(f"=== Finished dataset {id} ===\n")