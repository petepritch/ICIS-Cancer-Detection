# BoostedClassifier.py

import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report

class BoostedClassifier:
    def __init__(self, image_models, params=None, num_boost_round=500):
        """
        image_models: A list of ImageModel instances (each with extract_embeddings()).
        """
        self.image_models = image_models
        self.num_boost_round = num_boost_round
        self.params = params if params else {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'device': 'cuda',
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
        self.model = None

    def _build_dataset(self, loader):
        """
        Extract embeddings from each model, optionally combine with metadata, and
        collect labels.
        """
        embedding_list = []
        label_list = []
        
        for batch in loader:
            # batch can be (images, labels) or (images, labels, meta)
            if len(batch) == 2:
                images, labels = batch
                meta = None
            elif len(batch) == 3:
                images, labels, meta = batch
            else:
                raise ValueError("Unexpected batch size from dataset. Expected 2 or 3 elements.")
            
            # Extract embeddings from each model
            model_embeddings = []
            for im in self.image_models:
                emb = im.extract_embeddings(images).detach().cpu().numpy()
                model_embeddings.append(emb)
            
            # Concatenate embeddings horizontally
            combined_embeddings = np.hstack(model_embeddings)
            
            # If metadata is present, combine it too
            if meta is not None:
                meta_np = meta.detach().cpu().numpy()
                # shape: (batch_size, # metadata features)
                combined_embeddings = np.hstack([combined_embeddings, meta_np])
            
            embedding_list.append(combined_embeddings)
            label_list.append(labels.numpy())
        
        X = np.vstack(embedding_list)
        y = np.concatenate(label_list)
        return X, y

    def train_from_loader(self, train_loader):
        X_train, y_train = self._build_dataset(train_loader)
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        self.params['scale_pos_weight'] = num_neg / num_pos if num_pos > 0 else 1
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dtrain, 'train')],
            verbose_eval=10
        )

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        y_pred_probs = self.model.predict(dtest)
        return (y_pred_probs > 0.5).astype(int)

    def evaluate_from_loader(self, test_loader):
        X_test, y_test = self._build_dataset(test_loader)
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)
        return report

    def save_model(self, filepath):
        if self.model:
            self.model.save_model(filepath)

    def load_model(self, filepath):
        self.model = xgb.Booster()
        self.model.load_model(filepath)
