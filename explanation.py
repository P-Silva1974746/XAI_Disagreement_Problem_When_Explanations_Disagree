import openml
import sys
import joblib
import logging
import numpy as np
import pandas as pd
import shap
import lime
import gc
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
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

def get_feature_names(pipeline):
    preprocessor = pipeline.named_steps["preprocess"]
    return preprocessor.get_feature_names_out()

def shap_lr(model, X_background, X_explain):
    preprocessor = model.named_steps["preprocess"]
    lr_model = model.named_steps["clf"]

    X_bg_t = preprocessor.transform(X_background)
    X_ex_t = preprocessor.transform(X_explain)

    explainer = shap.LinearExplainer(
        lr_model,
        X_bg_t,
        feature_perturbation="interventional"
    )
    return explainer.shap_values(X_ex_t)

def shap_rf(model, X_explain):
    preprocessor = model.named_steps["preprocess"]
    rf_model = model.named_steps["clf"]

    X_transformed = preprocessor.transform(X_explain)

    # Shap can't handle sparse input so it is necessary to make it dense
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # Shap also needs the input to numeric  Ensure numeric dtype
    X_transformed = X_transformed.astype(np.float64)

    explainer = shap.TreeExplainer(rf_model)
    return explainer.shap_values(X_transformed)

def shap_mlp(model, X_background, X_explain, nsamples=100):
    feature_names = X_background.columns

    def predict_fn(x):
        # KernelSHAP gives numpy arrays it is needed convert back to DataFrame so that our pipeline can use the ColumnTransformer
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=feature_names)
        return model.predict_proba(x)

    background = shap.sample(X_background, nsamples)

    explainer = shap.KernelExplainer(
        predict_fn,
        background
    )

    shap_values = explainer.shap_values(X_explain.values, nsamples=nsamples)
    return shap_values

def build_categorical_imputer(model):
    """Creates a categorical Nan imputer"""
    preprocess = model.named_steps["preprocess"]
    cat_transformer = dict(preprocess.named_transformers_)["cat"]

    # categories learned during fit
    categories = cat_transformer.categories_

    # use first category as a safe fill value
    fill_values = {
        col: cats[0]
        for col, cats in zip(preprocess.transformers_[1][2], categories)
    }
    return fill_values

def lime_explainer(model, X_train):
    preprocess = model.named_steps["preprocess"]
    X_train_trans = preprocess.transform(X_train)

    if hasattr(X_train_trans, "toarray"):
        X_train_trans = X_train_trans.toarray()

    X_train_trans = X_train_trans.astype(np.float32)

    feature_names = preprocess.get_feature_names_out()

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_trans,
        feature_names=feature_names,
        class_names=["class_0", "class_1"],
        mode="classification",
        discretize_continuous=False,
        random_state=42
    )
    return explainer

def lime_explain_instance(explainer, model, x_instance, feature_names, cat_fill_values):

    if cat_fill_values is not None:

        preprocess = model.named_steps["preprocess"]

        x_df = pd.DataFrame([x_instance], columns=feature_names)

        # manually impute categorical NaNs to fix fragile lime interaction with Nan values
        for col, fill in cat_fill_values.items():
            if col in x_df.columns:
                x_df[col] = x_df[col].fillna(fill)

        x_trans = preprocess.transform(x_df)

        if hasattr(x_trans, "toarray"):
            x_trans = x_trans.toarray()

        x_trans = x_trans.astype(np.float32)

        def predict_fn(x):
            return model.named_steps["clf"].predict_proba(x.astype(np.float32))

        exp = explainer.explain_instance(
            x_trans[0],
            predict_fn,
            num_features=x_trans.shape[1]
        )

        return exp.as_list()
    
    else:

        preprocess = model.named_steps["preprocess"] 
        x_df = pd.DataFrame([x_instance], columns=feature_names) 
        x_trans = preprocess.transform(x_df) 
        
        if hasattr(x_trans, "toarray"): 
            x_trans = x_trans.toarray() 
        
        def predict_fn(x): 
            x = x.astype(np.float32) 
            return model.named_steps["clf"].predict_proba(x) 
        
        exp = explainer.explain_instance(
            x_trans[0], 
            predict_fn, 
            num_features=x_trans.shape[1], ) 
        
        return exp.as_list()
    
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

def lime_to_matrix(lime_exps, encoded_feature_names):
    
    n_samples = len(lime_exps)
    n_features = len(encoded_feature_names)

    mat = np.zeros((n_samples, n_features))
    name_to_idx = {n: i for i, n in enumerate(encoded_feature_names)}

    for i, lime_exp in enumerate(lime_exps):
        for name, val in lime_exp:
            # LIME uses "feature=value" or "feature <= x"
            clean = (
                name.replace("=", "_")
                    .replace("<=", "_")
                    .replace(">", "_")
                    .replace(" ", "")
            )

            if clean in name_to_idx:
                mat[i, name_to_idx[clean]] = abs(val)

    return mat

def to_long_df(values, features, model, explainer):

    n_runs, n_features = values.shape
    assert n_features == len(features)

    df = pd.DataFrame(
        values,
        columns=features
    )

    df["model"] = model
    df["explainer"] = explainer
    return df



if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("explanations_SHAP_LIME.log", mode="w"),
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

            # Background & explanation sets
            X_background = X_train.sample(100, random_state=42)
            X_explain = X_test.iloc[:50]


            # SHAP
            shap_lr_vals = shap_lr(models["lr"], X_background, X_explain)
            print("\tLR Shap calculated")
            shap_rf_vals = shap_rf(models["rf"], X_explain)
            print("\tRF Shap calculated")
            shap_mlp_vals = shap_mlp(models["mlp"], X_background, X_explain)
            print("\tMLP Shap calculated")



            lime_lr_vals=[]
            lime_rf_vals=[]
            lime_mlp_vals=[]

            # LIME
            lime_exp = lime_explainer(
                model=models["lr"],
                X_train=X_train,
            )

            cat_fill_values = None
            for val in categorical_mask:
                if val:
                    cat_fill_values = build_categorical_imputer(models["lr"])
                    break
                
            for i in range(50):
                try:
                    lime_lr_vals.append(lime_explain_instance(
                        lime_exp,
                        models["lr"],
                        X_explain.iloc[i].values,
                        col_names,
                        cat_fill_values
                    ))
                    if i==49:
                        logger.info("\tLR Lime calculated")

                    lime_rf_vals.append(lime_explain_instance(
                        lime_exp,
                        models["rf"],
                        X_explain.iloc[i].values,
                        col_names,
                        cat_fill_values
                    ))
                    if i==49:
                        logger.info("\tRF Lime calculated")

                    lime_mlp_vals.append(lime_explain_instance(
                        lime_exp,
                        models["mlp"],
                        X_explain.iloc[i].values,
                        col_names,
                        cat_fill_values
                    ))
                    if i==49:
                        logger.info("\tMLP Lime calculated")
                    if i%5==0:
                        logger.info(f"\tLime calulated for instance: {i}")

                except Exception as e:
                    logger.info(f"Instance: {i}")
                    logger.info(e)

            
            #Logistic Regression
            encoded_feature_names = get_feature_names(pipeline=models['lr'])
            feature_groups = build_feature_groups(encoded_feature_names, colname, categorical_mask)

            agg_shap_vals_lr, agg_feature_names = aggregate_attributions(
                shap_lr_vals,
                feature_groups
            )
            logger.info(f"Logistic Regression aggregated shap values calculated new shape: {agg_shap_vals_lr.shape}")

            lime_lr_matrix = lime_to_matrix(lime_lr_vals, encoded_feature_names)
            agg_lime_lr_vals, agg_feature_names = aggregate_attributions(
                lime_lr_matrix,
                feature_groups
            )
            logger.info(f"Logistic Regression aggregated lime values calculated new shape: {agg_lime_lr_vals.shape}")

            # Random Forest

            encoded_feature_names = get_feature_names(pipeline=models['rf'])
            feature_groups = build_feature_groups(encoded_feature_names, colname, categorical_mask)

            agg_shap_vals_rf, agg_feature_names = aggregate_attributions(
                shap_rf_vals[:,:,1], #only class one shap values
                feature_groups
            )
            logger.info(f"Random Forest aggregated shap values calculated new shape: {agg_shap_vals_rf.shape}")


            lime_rf_matrix = lime_to_matrix(lime_rf_vals, encoded_feature_names)
            agg_lime_rf_vals, agg_feature_names = aggregate_attributions(
                lime_rf_matrix,
                feature_groups
            )
            logger.info(f"Random Fores aggregated lime values calculated new shape: {agg_lime_rf_vals.shape}")


            # MLP
            agg_shap_vals_mlp = np.abs(shap_mlp_vals[:, :, 1])
            agg_feature_names = X_explain.columns.tolist()
            logger.info(f"MLP aggregated shap values calculated new shape: {agg_shap_vals_mlp.shape}")

            lime_mlp_matrix = lime_to_matrix(lime_mlp_vals, encoded_feature_names)
            agg_lime_mlp_vals, agg_feature_names = aggregate_attributions(
                lime_mlp_matrix,
                feature_groups
            )
            logger.info(f"MLP aggregated lime values calculated new shape: {agg_lime_mlp_vals.shape}")


            agg_mean_shap_vals_lr=[]
            agg_mean_shap_vals_rf=[]
            agg_mean_shap_vals_mlp=[]

            agg_mean_lime_vals_lr=[]
            agg_mean_lime_vals_rf=[]
            agg_mean_lime_vals_mlp=[]

            for i in range(agg_shap_vals_lr.shape[1]):
                agg_mean_shap_vals_lr.append(agg_shap_vals_lr[:,i].mean())
                agg_mean_shap_vals_rf.append(agg_shap_vals_rf[:,i].mean())
                agg_mean_shap_vals_mlp.append(agg_shap_vals_mlp[:,i].mean())

                agg_mean_lime_vals_lr.append(agg_lime_lr_vals[:,i].mean())
                agg_mean_lime_vals_rf.append(agg_lime_rf_vals[:,i].mean())
                agg_mean_lime_vals_mlp.append(agg_lime_mlp_vals[:,i].mean())


            dfs = [
                to_long_df(agg_shap_vals_lr,  colname, "lr",  "shap"),
                to_long_df(agg_shap_vals_rf,  colname, "rf",  "shap"),
                to_long_df(agg_shap_vals_mlp, colname, "mlp", "shap"),
                to_long_df(agg_lime_lr_vals,  colname, "lr",  "lime"),
                to_long_df(agg_lime_rf_vals,  colname, "rf",  "lime"),
                to_long_df(agg_lime_mlp_vals, colname, "mlp", "lime"),
            ]

            final_df = pd.concat(dfs, ignore_index=True)


            final_df.to_csv(f"results/SHAP_LIME_{id}.csv")
            logger.info(f"Saved results")
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
                "models", "X_explain", "shap_lr_vals", "lime_lr_matrix",
                "shap_rf_vals", "lime_rf_matrix", "shap_mlp_vals",
                "agg_shap_vals_lr", "agg_lime_lr_vals", "agg_shap_vals_rf",
                "agg_lime_rf_vals", "agg_shap_vals_mlp",
                "encoded_feature_names", "feature_groups", "agg_ig_vals_mlp",
                "agg_feature_names", "final_df",
            ]:
                if var in locals():
                    del locals()[var]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

            logger.info(f"=== Finished dataset {id} ===\n")

