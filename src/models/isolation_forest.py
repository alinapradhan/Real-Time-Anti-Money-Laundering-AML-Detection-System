import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any
import pickle
import os
from loguru import logger

class IsolationForestDetector:
    """Isolation Forest-based anomaly detection for financial transactions"""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize Isolation Forest detector
        
        Args:
            contamination: Expected proportion of outliers in the data
            random_state: Random state for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for anomaly detection
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame()
        
        # Basic transaction features
        features['amount'] = df['amount']
        features['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        features['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Account-based features
        features['source_account_encoded'] = pd.Categorical(df['source_account']).codes
        
        # Transaction type encoding
        transaction_type_map = {'deposit': 0, 'withdrawal': 1, 'transfer': 2, 'payment': 3}
        features['transaction_type_encoded'] = df['transaction_type'].map(transaction_type_map)
        
        # Channel encoding
        channel_map = {'online': 0, 'atm': 1, 'branch': 2, 'mobile': 3}
        features['channel_encoded'] = df['channel'].map(channel_map).fillna(0)
        
        # Amount-based features
        features['amount_log'] = np.log1p(features['amount'])
        
        # Handle missing values
        features = features.fillna(0)
        
        return features
    
    def fit(self, df: pd.DataFrame) -> 'IsolationForestDetector':
        """
        Train the Isolation Forest model
        
        Args:
            df: Training data DataFrame
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training Isolation Forest with {len(df)} transactions")
        
        # Extract features
        features = self._extract_features(df)
        self.feature_names = features.columns.tolist()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled)
        self.is_fitted = True
        
        logger.info("Isolation Forest training completed")
        return self
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies in transactions
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Tuple of (predictions, anomaly_scores)
            predictions: -1 for anomalies, 1 for normal
            anomaly_scores: Lower scores indicate higher anomaly likelihood
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features
        features = self._extract_features(df)
        
        # Ensure same features as training
        if list(features.columns) != self.feature_names:
            logger.warning("Feature names don't match training data")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions = self.model.predict(features_scaled)
        anomaly_scores = self.model.score_samples(features_scaled)
        
        return predictions, anomaly_scores
    
    def get_anomaly_probability(self, anomaly_scores: np.ndarray) -> np.ndarray:
        """
        Convert anomaly scores to probabilities
        
        Args:
            anomaly_scores: Raw anomaly scores from the model
            
        Returns:
            Probabilities (0-1) where higher values indicate higher anomaly likelihood
        """
        # Normalize scores to 0-1 range (invert since lower scores = higher anomaly)
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        
        if max_score == min_score:
            return np.zeros_like(anomaly_scores)
        
        # Invert and normalize
        probabilities = 1 - (anomaly_scores - min_score) / (max_score - min_score)
        return probabilities
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'contamination': self.contamination,
            'random_state': self.random_state
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'IsolationForestDetector':
        """Load a trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.contamination = model_data['contamination']
        self.random_state = model_data['random_state']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return self