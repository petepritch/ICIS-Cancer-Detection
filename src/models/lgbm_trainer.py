import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import logging

logger = logging.getLogger(__name__)


class LGBMModelTrainer:
    """
    Class for training and evaluating LightGBM models using image embeddings and metadata
    """

    def __init__(self, config):
        """
        Initialize the LGBM model trainer

        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config
        self.random_seed = config.get("random_seed", 42)
        self.n_trials = config.get("n_trials", 100)
        self.embedding_dir = config.get("embedding_dir")
        self.model_save_dir = config.get("model_save_dir")
        self.model_names = config.get("model_names", [])
        self.id_col = config.get("id_column", "isic_id")
        self.target_col = config.get("target_column", "target")
        self.test_size = config.get("test_size", 0.2)
        self.undersample_ratio = config.get("undersample_ratio", 0.01)
        self.oversample_ratio = config.get("oversample_ratio", None)
        self.n_jobs = config.get("n_jobs", -1)

        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Initialize variables that will be set later
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.best_model = None
        self.feature_importance = None
        self.study = None

    def load_and_prepare_data(self, metadata_path):
        """
        Load metadata and embedding CSVs, merge them, and prepare train/test splits

        Args:
            metadata_path: Path to metadata CSV file

        Returns:
            X_train, y_train, X_test, y_test
        """
        logger.info("Loading metadata from %s", metadata_path)

        # Load metadata
        try:
            df_meta = pd.read_csv(metadata_path)
            df_meta[self.id_col] = df_meta[self.id_col].astype(str)
            logger.info(f"Loaded metadata with shape: {df_meta.shape}")

            # Create train/test split based on metadata
            meta_indices = df_meta.index
            meta_targets = df_meta[self.target_col].values
            train_meta_idx, test_meta_idx = train_test_split(
                meta_indices,
                test_size=self.test_size,
                random_state=self.random_seed,
                stratify=meta_targets,
            )

            df_meta_train = df_meta.loc[train_meta_idx].copy()
            df_meta_test = df_meta.loc[test_meta_idx].copy()
            logger.info(
                f"Split metadata: Train shape={df_meta_train.shape}, Test shape={df_meta_test.shape}"
            )

            # Initialize final dataframes with metadata
            df_train_final = df_meta_train.copy()
            df_test_final = df_meta_test.copy()

            # Merge embeddings from each model
            for model_name in self.model_names:
                logger.info(f"Processing embeddings for model: {model_name}")

                train_embed_path = os.path.join(
                    self.embedding_dir, f"train_embeddings_{model_name}.csv"
                )
                test_embed_path = os.path.join(
                    self.embedding_dir, f"test_embeddings_{model_name}.csv"
                )

                if not os.path.exists(train_embed_path):
                    raise FileNotFoundError(
                        f"Train embedding file not found: {train_embed_path}"
                    )
                if not os.path.exists(test_embed_path):
                    raise FileNotFoundError(
                        f"Test embedding file not found: {test_embed_path}"
                    )

                # Load embeddings
                df_train_embed = pd.read_csv(train_embed_path)
                df_test_embed = pd.read_csv(test_embed_path)

                # Ensure ID column is string type
                df_train_embed[self.id_col] = df_train_embed[self.id_col].astype(str)
                df_test_embed[self.id_col] = df_test_embed[self.id_col].astype(str)

                # Merge with accumulated data
                df_train_final = pd.merge(
                    df_train_final,
                    df_train_embed.drop(columns=[self.target_col], errors="ignore"),
                    on=self.id_col,
                    how="inner",
                )

                df_test_final = pd.merge(
                    df_test_final,
                    df_test_embed.drop(columns=[self.target_col], errors="ignore"),
                    on=self.id_col,
                    how="inner",
                )

                logger.info(
                    f"  After merging {model_name}: Train shape={df_train_final.shape}, Test shape={df_test_final.shape}"
                )

            # Verify merge results
            if len(df_train_final) != len(df_meta_train):
                logger.warning(
                    f"Train data row count mismatch! Original={len(df_meta_train)}, Merged={len(df_train_final)}"
                )
            if len(df_test_final) != len(df_meta_test):
                logger.warning(
                    f"Test data row count mismatch! Original={len(df_meta_test)}, Merged={len(df_test_final)}"
                )

            # Prepare X and y
            self.y_train = df_train_final[self.target_col]
            self.y_test = df_test_final[self.target_col]

            # Get feature columns (all except ID and target)
            feature_cols = [
                col
                for col in df_train_final.columns
                if col not in [self.id_col, self.target_col]
            ]

            if not feature_cols:
                raise ValueError("No feature columns identified after merge.")

            self.X_train = df_train_final[feature_cols]
            self.X_test = df_test_final[feature_cols]

            # Store IDs for later use
            self.train_ids = df_train_final[self.id_col].values
            self.test_ids = df_test_final[self.id_col].values

            # Log dataset details
            num_meta_cols = len(df_meta.columns) - 2  # Exclude ID and Target
            num_embed_cols = len(feature_cols) - num_meta_cols
            logger.info(
                f"Selected {len(feature_cols)} total features ({num_embed_cols} embeddings + {num_meta_cols} metadata)."
            )
            logger.info(
                f"Final dataset shapes: X_train={self.X_train.shape}, X_test={self.X_test.shape}"
            )

            return self.X_train, self.y_train, self.X_test, self.y_test

        except Exception as e:
            logger.exception(f"Error in data preparation: {e}")
            raise

    def _pauc_scorer(self, y_true, y_pred_proba, min_tpr=0.8):
        """Calculate partial AUC focused on high sensitivity region"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

        # Find points above min_tpr
        high_sens_idx = np.where(tpr >= min_tpr)[0]

        if len(high_sens_idx) == 0:
            return 0.0

        # Get minimum index where TPR >= min_tpr
        start_idx = high_sens_idx[0]

        # Handle interpolation for exact threshold
        if start_idx > 0:
            prev_idx = start_idx - 1
            slope = (fpr[start_idx] - fpr[prev_idx]) / (tpr[start_idx] - tpr[prev_idx])
            interp_fpr = fpr[prev_idx] + slope * (min_tpr - tpr[prev_idx])

            high_sens_fpr = np.concatenate(([interp_fpr], fpr[start_idx:]))
            high_sens_tpr = np.concatenate(([min_tpr], tpr[start_idx:]))
        else:
            high_sens_fpr = fpr[start_idx:]
            high_sens_tpr = tpr[start_idx:]

        # Calculate pAUC using trapezoidal rule
        pauc = np.trapz(1 - high_sens_fpr, high_sens_tpr)

        # Normalize by maximum possible area (1 - min_tpr)
        norm_pauc = pauc / (1 - min_tpr)

        return norm_pauc

    def _lgbm_pauc_eval_metric(self, y_true, y_pred_proba):
        """Format pAUC for LightGBM evaluation metric"""
        pauc_value = self._pauc_scorer(y_true, y_pred_proba)
        return "pAUC", pauc_value, True  # True means higher is better

    def _optuna_objective(self, trial, x_train, y_train, x_val, y_val):
        """Objective function for Optuna hyperparameter optimization"""
        # Apply resampling if specified
        if self.undersample_ratio:
            sampler = RandomUnderSampler(
                sampling_strategy=self.undersample_ratio, random_state=self.random_seed
            )
            x_train_resampled, y_train_resampled = sampler.fit_resample(
                x_train, y_train
            )
        else:
            x_train_resampled, y_train_resampled = x_train, y_train

        if self.oversample_ratio:
            oversampler = RandomOverSampler(
                sampling_strategy=self.oversample_ratio, random_state=self.random_seed
            )
            x_train_resampled, y_train_resampled = oversampler.fit_resample(
                x_train_resampled, y_train_resampled
            )

        # Define hyperparameter search space
        lgbm_params = {
            "objective": "binary",
            "metric": "auc",  # Use AUC for internal LGBM monitoring
            "verbosity": -1,
            "random_state": self.random_seed,
            "n_estimators": 2000,  # High value, rely on early stopping
            "n_jobs": self.n_jobs,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 4.0),
        }

        # Train model for this trial
        model = lgb.LGBMClassifier(**lgbm_params)

        # Fit with early stopping
        model.fit(
            x_train_resampled,
            y_train_resampled,
            eval_set=[(x_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )

        # Use pAUC to evaluate trial
        y_pred_proba_val = model.predict_proba(x_val)[:, 1]
        val_pauc = self._pauc_scorer(y_val, y_pred_proba_val)

        logger.info(
            f"Trial {trial.number}: Val pAUC={val_pauc:.6f} (at iter {model.best_iteration_})"
        )

        return val_pauc

    def tune_hyperparameters(self):
        """Run Optuna hyperparameter optimization"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call load_and_prepare_data first.")

        logger.info(f"Starting Optuna optimization with {self.n_trials} trials")
        study_name = f"lgbm_tuning_{time.strftime('%Y%m%d_%H%M%S')}"

        self.study = optuna.create_study(direction="maximize", study_name=study_name)

        # Prepare objective function with data
        objective_with_data = lambda trial: self._optuna_objective(
            trial, self.X_train, self.y_train, self.X_test, self.y_test
        )

        start_time = time.time()
        try:
            self.study.optimize(objective_with_data, n_trials=self.n_trials)
        except Exception as e:
            logger.exception(f"Error during Optuna optimization: {e}")
            raise
        finally:
            elapsed_time = time.time() - start_time
            logger.info(f"Optimization finished in {elapsed_time:.2f} seconds")

        # Log best trial results
        logger.info(f"Best trial: {self.study.best_trial.number}")
        logger.info(f"Best pAUC: {self.study.best_value:.6f}")
        logger.info("Best hyperparameters:")
        for key, value in self.study.best_params.items():
            logger.info(f"  {key}: {value}")

        return self.study.best_params

    def train_final_model(self, best_params=None):
        """Train final model with either provided or tuned parameters"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call load_and_prepare_data first.")

        # Get parameters for final model
        if best_params:
            final_params = best_params.copy()
        elif self.study and self.study.best_params:
            final_params = self.study.best_params.copy()
        else:
            logger.warning(
                "No best parameters available. Using defaults with calculated weight."
            )
            neg_count = (self.y_train == 0).sum()
            pos_count = (self.y_train == 1).sum()
            pos_weight = neg_count / pos_count if pos_count > 0 else 1

            final_params = {
                "objective": "binary",
                "metric": "auc",
                "random_state": self.random_seed,
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "scale_pos_weight": pos_weight,
            }

        # Ensure basic parameters are set
        final_params["objective"] = "binary"
        final_params["random_state"] = self.random_seed
        final_params["n_jobs"] = self.n_jobs
        final_params["n_estimators"] = 5000  # We'll use early stopping

        logger.info("Final model parameters:")
        for key, value in final_params.items():
            logger.info(f"  {key}: {value}")

        # Create final model
        self.best_model = lgb.LGBMClassifier(**final_params)

        # Apply resampling if specified
        if self.undersample_ratio:
            sampler = RandomUnderSampler(
                sampling_strategy=self.undersample_ratio, random_state=self.random_seed
            )
            X_train_resampled, y_train_resampled = sampler.fit_resample(
                self.X_train, self.y_train
            )
        else:
            X_train_resampled, y_train_resampled = self.X_train, self.y_train

        if self.oversample_ratio:
            oversampler = RandomOverSampler(
                sampling_strategy=self.oversample_ratio, random_state=self.random_seed
            )
            X_train_resampled, y_train_resampled = oversampler.fit_resample(
                X_train_resampled, y_train_resampled
            )

        # Train final model with early stopping
        logger.info("Training final model...")
        start_time = time.time()

        eval_func = lambda y_true, y_pred: self._lgbm_pauc_eval_metric(y_true, y_pred)

        self.best_model.fit(
            X_train_resampled,
            y_train_resampled,
            eval_set=[(self.X_test, self.y_test)],
            eval_metric=eval_func,
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=100)],
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Final model training took {elapsed_time:.2f} seconds")
        logger.info(f"Best iteration: {self.best_model.best_iteration_}")

        # Store feature importance
        if hasattr(self.best_model, "feature_importances_"):
            self.feature_importance = pd.DataFrame(
                {
                    "Feature": self.X_train.columns,
                    "Importance": self.best_model.feature_importances_,
                }
            ).sort_values("Importance", ascending=False)

        return self.best_model

    def evaluate_model(self, threshold=0.5):
        """Evaluate the final model and return metrics"""
        if self.best_model is None:
            raise ValueError(
                "No trained model available. Call train_final_model first."
            )

        logger.info("Evaluating final model on test set")

        # Get predictions
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        y_pred_binary = (y_pred_proba > threshold).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred_binary)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        pauc_score = self._pauc_scorer(self.y_test, y_pred_proba)
        cm = confusion_matrix(self.y_test, y_pred_binary)

        # Extract specific metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Compile metrics
        metrics = {
            "accuracy": accuracy,
            "auc": auc_score,
            "pauc": pauc_score,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }

        # Log metrics
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test AUC: {auc_score:.6f}")
        logger.info(f"Test pAUC: {pauc_score:.6f}")
        logger.info(f"Test Sensitivity: {sensitivity:.4f}")
        logger.info(f"Test Specificity: {specificity:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")

        return metrics

    def save_model(self, filename=None):
        """Save the trained model to disk"""
        if self.best_model is None:
            raise ValueError(
                "No trained model available. Call train_final_model first."
            )

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_names_str = (
                "_".join([n.split("_")[0] for n in self.model_names])
                if self.model_names
                else "model"
            )
            filename = f"lgbm_{model_names_str}_{timestamp}.pkl"

        filepath = os.path.join(self.model_save_dir, filename)

        # Save model with joblib
        joblib.dump(self.best_model, filepath)
        logger.info(f"Model saved to: {filepath}")

        # Also save feature importance if available
        if self.feature_importance is not None:
            importance_path = os.path.join(
                self.model_save_dir, f"{os.path.splitext(filename)[0]}_importance.csv"
            )
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to: {importance_path}")

        return filepath

    def load_model(self, filepath):
        """Load a trained model from disk"""
        self.best_model = joblib.load(filepath)
        logger.info(f"Model loaded from: {filepath}")
        return self.best_model

    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve for the model"""
        if self.best_model is None or self.X_test is None or self.y_test is None:
            raise ValueError("Model or test data not available")

        # Get predictions for test set
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
        plt.plot([0, 1], [0, 1], "k--", lw=2)

        # Highlight pAUC region
        min_tpr = 0.8
        highsens_idx = np.where(tpr >= min_tpr)[0][0]
        plt.fill_between(
            fpr[highsens_idx:],
            tpr[highsens_idx:],
            alpha=0.3,
            color="blue",
            label=f"pAUC (TPR ≥ {min_tpr})",
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"ROC curve saved to: {save_path}")

        plt.show()

        return fpr, tpr, auc_score

    def plot_feature_importance(self, top_n=20, save_path=None):
        """Plot feature importance"""
        if self.feature_importance is None:
            raise ValueError("Feature importance not available")

        # Get top N features
        top_features = self.feature_importance.head(top_n)

        plt.figure(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=top_features)
        plt.title(f"Top {top_n} Feature Importance")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to: {save_path}")

        plt.show()

        return top_features
