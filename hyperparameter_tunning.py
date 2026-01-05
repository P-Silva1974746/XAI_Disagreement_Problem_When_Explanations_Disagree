import gc
import logging
import sys
import os
import random
import openml
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from scipy.stats import loguniform, randint
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping





class MLP_IG(nn.Module):
    def __init__(self, output_dim, hidden_dim=16):
        super().__init__()

        self.fc1 = nn.LazyLinear(hidden_dim)
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


# using a fucntion instead of a lambda function is necessary in order to be able to save the MLP, since joblib Can't pickle function <lambda>
def to_float32_fn(X):
    return X.astype(np.float32)



if __name__ == '__main__':

    os.makedirs("models", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("hyperparameter_tuning.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)


    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    datasets= [1504, 1590, 1510, 37, 31, 59, 45547, 44, 1558, 40981]

    for id in datasets:

        logger.info(f"\n=== Starting dataset {id} ===")

        try:
            dataset = openml.datasets.get_dataset(dataset_id=id, download_data=True, download_qualities=True, download_features_meta_data=True)
            X, y, categorical_mask, colname=dataset.get_data(target=dataset.default_target_attribute , dataset_format="dataframe")

            string_cols = X.select_dtypes(include=["object"]).columns.tolist()
            if string_cols:
                logger.info(f"Dropping string columns in dataset {id}: {string_cols}")
                X = X.drop(columns=string_cols)
            
            logger.info(f"{dataset.name} (id: {dataset.id})")

            n_classes = y.nunique()

            label_enc= LabelEncoder()
            y = label_enc.fit_transform(y=y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

            categorical_cols = X.select_dtypes(include=["category", "bool"]).columns.tolist()

            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ]
            )


            lr_pipe = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("clf", LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        tol=1e-4,
                        random_state=42,
                        solver="saga",
                    ))
                ]
            )

            lr_param_dist = {
                "clf__C": loguniform(1e-4, 1e2),
                "clf__l1_ratio": np.linspace(0, 1, 5)
            }


            lr_search = RandomizedSearchCV(
                estimator=lr_pipe,
                param_distributions=lr_param_dist,
                n_iter=30,
                scoring="balanced_accuracy",
                cv=5,
                random_state=42,
                n_jobs=8,
                verbose=1,
                return_train_score=True,
            )

            lr_search.fit(X_train, y_train)
            best_lr = lr_search.best_estimator_

            logger.info("\n\tBest Logistic Regression params:")
            logger.info(f"\t\t{lr_search.best_params_}")



            rf_pipe = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("clf", RandomForestClassifier(
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=8,
                    ))
                ]
            )

            rf_param_dist = {
                "clf__n_estimators": randint(300, 600),
                "clf__max_depth": [None, 10 ,20, 30],
                "clf__min_samples_split": randint(2, 20),
                "clf__min_samples_leaf": randint(1, 10),
                "clf__max_features": ["sqrt", "log2", None]
            }

            rf_search = RandomizedSearchCV(
                estimator=rf_pipe,
                param_distributions=rf_param_dist,
                n_iter=20,
                scoring="balanced_accuracy",
                cv=3,
                random_state=42,
                n_jobs=1,
                verbose=1,
                return_train_score=True,
            )


            rf_search.fit(X_train, y_train)
            best_rf = rf_search.best_estimator_

            logger.info("\tBest Random Forest params:")
            logger.info(f"\t\t{rf_search.best_params_}")



            net = NeuralNetClassifier(
                module=MLP_IG,
                module__output_dim=n_classes,
                criterion=nn.CrossEntropyLoss,
                optimizer=torch.optim.Adam,
                max_epochs=100,
                batch_size=64,
                iterator_train__shuffle=True,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=0,
                callbacks=[EarlyStopping(patience=10)],
            )

            # to resolve conflict error between the the preprocesser type output float64 and the 
            # skorch converts NumPy to Torch without dtype casting, and the MLP weights are are float32
            # creating a conflit
            to_float32 = FunctionTransformer(
                to_float32_fn,
                accept_sparse=True
            )


            mlp_pipe = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("to_float32", to_float32),
                    ("clf", net)
                ]
            )

            mlp_param_dist = {
                "clf__lr": loguniform(1e-4, 1e-2),
                "clf__max_epochs": [50, 100, 150],
                "clf__batch_size": [32, 64, 128],
                "clf__optimizer__weight_decay": loguniform(1e-6, 1e-3),
                "clf__optimizer__betas": [(0.9, 0.999), (0.9, 0.99)],
            }

            mlp_search = RandomizedSearchCV(
                estimator=mlp_pipe,
                param_distributions=mlp_param_dist,
                n_iter=20,
                scoring="balanced_accuracy",
                cv=3,
                random_state=42,
                verbose=1,
                n_jobs=1,
            )


            mlp_search.fit(X_train, y_train)
            best_mlp = mlp_search.best_estimator_

            logger.info("\tBest MLP params:")
            logger.info(f"\t\t{mlp_search.best_params_}")



            pred = best_lr.predict(X_test)
            logger.info("\n\tLogistic Regression performance:")
            logger.info("\n\n" + classification_report(y_test, pred))

            pred = best_rf.predict(X_test)
            logger.info("\tRandom Forest performance:")
            logger.info("\n\n" + classification_report(y_test, pred))

            pred = best_mlp.predict(X_test)
            logger.info("\tMLP performance:")
            logger.info("\n\n" + classification_report(y_test, pred))

            logger.info("\n\n\n#------------------------------------------------------------------------------#\n\n\n")

            # save bestmodels
            joblib.dump(best_lr, f"models/lr_{id}.joblib")
            joblib.dump(best_rf, f"models/rf_{id}.joblib")
            joblib.dump(best_mlp, f"models/mlp_{id}.joblib")
        
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
                "lr_search", "rf_search", "mlp_search",
                "best_lr", "best_rf", "best_mlp",
                "lr_pipe", "rf_pipe", "mlp_pipe",
                "preprocessor"
            ]:
                if var in locals():
                    del locals()[var]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

            logger.info(f"=== Finished dataset {id} ===\n")


