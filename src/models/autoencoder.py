import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional
import pickle
import os
from loguru import logger

class AutoencoderDetector:
    """Autoencoder-based anomaly detection for financial transactions"""
    
    def __init__(self, 
                 encoding_dim: int = 32,
                 threshold_percentile: float = 95,
                 epochs: int = 100,
                 batch_size: int = 32,
                 validation_split: float = 0.2):
        """
        Initialize Autoencoder detector
        
        Args:
            encoding_dim: Dimension of the encoded representation
            threshold_percentile: Percentile for reconstruction error threshold
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
        """
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        self.autoencoder = None
        self.encoder = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.reconstruction_threshold = None
        self.feature_names = None
        
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for autoencoder training
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame()
        
        # Temporal features
        dt = pd.to_datetime(df['timestamp'])
        features['hour'] = dt.dt.hour
        features['day_of_week'] = dt.dt.dayofweek
        features['month'] = dt.dt.month
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Amount features
        features['amount'] = df['amount']
        features['amount_log'] = np.log1p(df['amount'])
        features['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        # Categorical features (one-hot encoded)
        transaction_types = pd.get_dummies(df['transaction_type'], prefix='txn_type')
        features = pd.concat([features, transaction_types], axis=1)
        
        channels = pd.get_dummies(df['channel'], prefix='channel')
        features = pd.concat([features, channels], axis=1)
        
        # Account-based features
        features['source_account_hash'] = df['source_account'].astype(str).apply(hash) % 1000
        
        # Country features (simplified - in production, use proper encoding)
        features['source_country_hash'] = df['source_country'].astype(str).apply(hash) % 100
        
        if 'destination_country' in df.columns:
            features['dest_country_hash'] = df['destination_country'].fillna('Unknown').astype(str).apply(hash) % 100
        else:
            features['dest_country_hash'] = 0
        
        # Cross-border indicator
        if 'destination_country' in df.columns:
            features['cross_border'] = (df['source_country'] != df['destination_country'].fillna(df['source_country'])).astype(int)
        else:
            features['cross_border'] = 0
        
        # Handle missing values
        features = features.fillna(0)
        
        return features
    
    def _build_autoencoder(self, input_dim: int) -> None:
        """
        Build the autoencoder architecture
        
        Args:
            input_dim: Number of input features
        """
        # Encoder
        input_layer = keras.layers.Input(shape=(input_dim,))
        encoded = keras.layers.Dense(128, activation='relu')(input_layer)
        encoded = keras.layers.Dropout(0.2)(encoded)
        encoded = keras.layers.Dense(64, activation='relu')(encoded)
        encoded = keras.layers.Dropout(0.1)(encoded)
        encoded = keras.layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = keras.layers.Dense(64, activation='relu')(encoded)
        decoded = keras.layers.Dropout(0.1)(decoded)
        decoded = keras.layers.Dense(128, activation='relu')(decoded)
        decoded = keras.layers.Dropout(0.2)(decoded)
        decoded = keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Models
        self.autoencoder = keras.Model(input_layer, decoded)
        self.encoder = keras.Model(input_layer, encoded)
        
        # Compile
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Autoencoder architecture built with input dim: {input_dim}")
    
    def fit(self, df: pd.DataFrame) -> 'AutoencoderDetector':
        """
        Train the autoencoder model
        
        Args:
            df: Training data DataFrame
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training Autoencoder with {len(df)} transactions")
        
        # Extract features
        features = self._extract_features(df)
        self.feature_names = features.columns.tolist()
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Build model
        self._build_autoencoder(features_scaled.shape[1])
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = self.autoencoder.fit(
            features_scaled, features_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Calculate reconstruction threshold
        reconstructed = self.autoencoder.predict(features_scaled)
        reconstruction_errors = np.mean(np.square(features_scaled - reconstructed), axis=1)
        self.reconstruction_threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        
        self.is_fitted = True
        logger.info(f"Autoencoder training completed. Threshold: {self.reconstruction_threshold:.4f}")
        return self
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies using reconstruction error
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Tuple of (predictions, reconstruction_errors)
            predictions: 1 for anomalies, 0 for normal
            reconstruction_errors: Reconstruction error for each sample
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
        
        # Get reconstructions
        reconstructed = self.autoencoder.predict(features_scaled, verbose=0)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(features_scaled - reconstructed), axis=1)
        
        # Make predictions
        predictions = (reconstruction_errors > self.reconstruction_threshold).astype(int)
        
        return predictions, reconstruction_errors
    
    def get_anomaly_probability(self, reconstruction_errors: np.ndarray) -> np.ndarray:
        """
        Convert reconstruction errors to probabilities
        
        Args:
            reconstruction_errors: Raw reconstruction errors
            
        Returns:
            Probabilities (0-1) where higher values indicate higher anomaly likelihood
        """
        # Normalize using the threshold
        probabilities = np.minimum(reconstruction_errors / self.reconstruction_threshold, 2.0) / 2.0
        return probabilities
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        # Create directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save autoencoder model
        autoencoder_path = filepath.replace('.pkl', '_autoencoder.h5')
        self.autoencoder.save(autoencoder_path)
        
        # Save other components
        model_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'reconstruction_threshold': self.reconstruction_threshold,
            'encoding_dim': self.encoding_dim,
            'threshold_percentile': self.threshold_percentile
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'AutoencoderDetector':
        """Load a trained model from disk"""
        # Load autoencoder model
        autoencoder_path = filepath.replace('.pkl', '_autoencoder.h5')
        self.autoencoder = keras.models.load_model(autoencoder_path)
        
        # Load other components
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.reconstruction_threshold = model_data['reconstruction_threshold']
        self.encoding_dim = model_data['encoding_dim']
        self.threshold_percentile = model_data['threshold_percentile']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return self