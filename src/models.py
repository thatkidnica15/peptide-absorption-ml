"""
Permeability Prediction Models
================================
Three models of increasing complexity for predicting log(Papp).

Author: Veronica Pilagov
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.inspection import permutation_importance
import pandas as pd
from typing import Tuple


def train_gradient_boosting(X_train, y_train, feature_names):
    """
    Gradient Boosted Trees for permeability regression.
    
    GBT builds trees sequentially, each correcting the errors of
    the previous. Strong for tabular data with mixed feature types.
    """
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=10,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)
    
    return model, importance_df


def train_random_forest(X_train, y_train, feature_names):
    """Random Forest regressor with permutation importance."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)
    
    return model, importance_df


class NeuralNetRegressor:
    """
    Neural network for regression, implemented from scratch in NumPy.
    
    Architecture: Input -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Linear)
    Loss: Mean Squared Error
    Optimiser: Adam
    """
    
    def __init__(self, n_features, lr=0.001, seed=42):
        rng = np.random.default_rng(seed)
        
        # He initialisation
        self.W1 = rng.normal(0, np.sqrt(2/n_features), (n_features, 64))
        self.b1 = np.zeros((1, 64))
        self.W2 = rng.normal(0, np.sqrt(2/64), (64, 32))
        self.b2 = np.zeros((1, 32))
        self.W3 = rng.normal(0, np.sqrt(1/32), (32, 1))
        self.b3 = np.zeros((1, 1))
        
        self.lr = lr
        self.history = {'train_loss': [], 'val_loss': []}
        
        # Adam state
        self._init_adam()
    
    def _init_adam(self):
        self.t = 0
        self.m = {}
        self.v = {}
        for name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            param = getattr(self, name)
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)
    
    def _adam_update(self, name, grad, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad
        self.v[name] = beta2 * self.v[name] + (1 - beta2) * grad**2
        m_hat = self.m[name] / (1 - beta1**self.t)
        v_hat = self.v[name] / (1 - beta2**self.t)
        param = getattr(self, name)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
    
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(0, self.z2)  # ReLU
        self.z3 = self.a2 @ self.W3 + self.b3  # Linear output
        return self.z3
    
    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        
        # Output gradient (MSE derivative)
        dz3 = (2 / m) * (y_pred - y_true)
        dW3 = self.a2.T @ dz3
        db3 = dz3.sum(axis=0, keepdims=True)
        
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (self.z2 > 0)
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=300, batch_size=32, verbose=True):
        y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        if y_val is not None:
            y_val = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
        
        m = X_train.shape[0]
        rng = np.random.default_rng(42)
        
        for epoch in range(1, epochs + 1):
            self.t += 1
            idx = rng.permutation(m)
            
            epoch_loss = 0
            n_batches = 0
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                xb = X_train[idx[start:end]]
                yb = y_train[idx[start:end]]
                
                pred = self.forward(xb)
                loss = np.mean((pred - yb) ** 2)
                epoch_loss += loss
                n_batches += 1
                
                grads = self.backward(xb, yb, pred)
                for name in grads:
                    self._adam_update(name, grads[name])
            
            self.history['train_loss'].append(epoch_loss / n_batches)
            
            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = np.mean((val_pred - y_val) ** 2)
                self.history['val_loss'].append(val_loss)
            
            if verbose and epoch % 50 == 0:
                msg = f"Epoch {epoch}/{epochs} | Train MSE: {epoch_loss/n_batches:.4f}"
                if X_val is not None:
                    msg += f" | Val MSE: {val_loss:.4f}"
                print(msg)
        
        return self.history
    
    def predict(self, X):
        return self.forward(X).flatten()


def cross_validate(model_class, X, y, n_folds=5, **kwargs):
    """K-fold cross-validation for any sklearn-compatible model."""
    if hasattr(model_class, 'fit') and hasattr(model_class, 'predict'):
        model = model_class
    else:
        model = model_class(**kwargs)
    
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    
    return {
        'r2_mean': round(r2_scores.mean(), 4),
        'r2_std': round(r2_scores.std(), 4),
        'mae_mean': round(mae_scores.mean(), 4),
        'mae_std': round(mae_scores.std(), 4),
    }
