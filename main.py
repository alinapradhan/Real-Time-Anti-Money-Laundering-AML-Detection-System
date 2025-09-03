#!/usr/bin/env python3
"""
Real-Time Anti-Money Laundering (AML) Detection System
Main application entry point
"""

import argparse
import sys
import os
from datetime import datetime
from loguru import logger
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import AutoencoderDetector
from src.rules.rule_engine import RuleBasedAMLDetector
from src.pipeline.streaming import StreamingPipeline, KafkaTransactionProducer
from src.alerts.alert_manager import AlertManager
from src.utils.data_generator import TransactionGenerator, generate_sample_data
from config.settings import *

class AMLDetectionSystem:
    """Main AML Detection System orchestrator"""
    
    def __init__(self):
        """Initialize the AML Detection System"""
        self.isolation_forest = None
        self.autoencoder = None
        self.rule_detector = None
        self.alert_manager = None
        self.streaming_pipeline = None
        
        # Setup logging
        logger.add(LOG_FILE, rotation="100 MB", retention="30 days", level=LOG_LEVEL)
        logger.info("AML Detection System initializing...")
    
    def setup_models(self, train_models: bool = True):
        """Setup and train ML models"""
        logger.info("Setting up detection models...")
        
        # Initialize models
        self.isolation_forest = IsolationForestDetector(
            contamination=ISOLATION_FOREST_CONTAMINATION
        )
        self.autoencoder = AutoencoderDetector(
            threshold_percentile=AUTOENCODER_THRESHOLD * 100
        )
        self.rule_detector = RuleBasedAMLDetector()
        
        if train_models:
            # Generate training data
            logger.info("Generating training data...")
            generator = TransactionGenerator()
            training_transactions = generator.generate_normal_transactions(10000)
            
            # Convert to DataFrame
            training_df = pd.DataFrame([txn.dict() for txn in training_transactions])
            
            # Train models
            logger.info("Training Isolation Forest...")
            self.isolation_forest.fit(training_df)
            
            logger.info("Training Autoencoder...")
            self.autoencoder.fit(training_df)
            
            # Save models
            os.makedirs("models", exist_ok=True)
            self.isolation_forest.save_model("models/isolation_forest.pkl")
            self.autoencoder.save_model("models/autoencoder.pkl")
            
            logger.info("Models trained and saved successfully")
        else:
            # Load pre-trained models
            logger.info("Loading pre-trained models...")
            if os.path.exists("models/isolation_forest.pkl"):
                self.isolation_forest.load_model("models/isolation_forest.pkl")
            if os.path.exists("models/autoencoder.pkl"):
                self.autoencoder.load_model("models/autoencoder.pkl")
    
    def setup_pipeline(self):
        """Setup streaming pipeline and alert system"""
        logger.info("Setting up streaming pipeline...")
        
        # Initialize alert manager
        self.alert_manager = AlertManager()
        
        # Initialize streaming pipeline
        self.streaming_pipeline = StreamingPipeline(
            kafka_servers=KAFKA_BOOTSTRAP_SERVERS,
            transaction_topic=KAFKA_TRANSACTION_TOPIC,
            alert_topic=KAFKA_ALERT_TOPIC
        )
        
        # Set detectors
        self.streaming_pipeline.set_detectors(
            ml_detector=self.isolation_forest,
            rule_detector=self.rule_detector
        )
        
        # Connect alert manager to pipeline
        original_process = self.streaming_pipeline.process_transaction
        
        def enhanced_process(transaction):
            alerts = original_process(transaction)
            # Process alerts through alert manager
            if hasattr(alerts, '__iter__'):
                for alert in alerts:
                    self.alert_manager.process_alert(alert)
        
        self.streaming_pipeline.process_transaction = enhanced_process
        
        logger.info("Pipeline setup completed")
    
    def start_streaming(self):
        """Start the real-time streaming detection"""
        logger.info("Starting real-time AML detection...")
        
        if not self.streaming_pipeline:
            raise RuntimeError("Pipeline not setup. Call setup_pipeline() first.")
        
        self.streaming_pipeline.start()
        logger.info("Streaming detection started")
    
    def stop_streaming(self):
        """Stop the streaming detection"""
        if self.streaming_pipeline:
            self.streaming_pipeline.stop()
            logger.info("Streaming detection stopped")
    
    def run_batch_analysis(self, data_file: str):
        """Run batch analysis on a data file"""
        logger.info(f"Running batch analysis on {data_file}")
        
        # Load data
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        elif data_file.endswith('.json'):
            df = pd.read_json(data_file)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        # Run detection
        results = []
        
        if self.isolation_forest and self.isolation_forest.is_fitted:
            predictions, scores = self.isolation_forest.predict(df)
            anomaly_probs = self.isolation_forest.get_anomaly_probability(scores)
            
            for i, (pred, score, prob) in enumerate(zip(predictions, scores, anomaly_probs)):
                if pred == -1:  # Anomaly detected
                    results.append({
                        'transaction_index': i,
                        'detection_method': 'isolation_forest',
                        'anomaly_score': score,
                        'anomaly_probability': prob
                    })
        
        logger.info(f"Batch analysis completed. Found {len(results)} anomalies.")
        return results
    
    def generate_test_data(self, output_dir: str = "data"):
        """Generate test data for the system"""
        logger.info("Generating test data...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        generator = TransactionGenerator()
        
        # Generate normal transactions
        normal_txns = generator.generate_normal_transactions(5000)
        generator.save_transactions_to_csv(normal_txns, f"{output_dir}/normal_transactions.csv")
        
        # Generate suspicious transactions
        suspicious_txns = generator.generate_suspicious_transactions(500, "mixed")
        generator.save_transactions_to_csv(suspicious_txns, f"{output_dir}/suspicious_transactions.csv")
        
        # Generate mixed dataset
        all_txns = normal_txns + suspicious_txns
        generator.save_transactions_to_csv(all_txns, f"{output_dir}/mixed_transactions.csv")
        
        logger.info(f"Test data generated in {output_dir}")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Real-Time AML Detection System")
    parser.add_argument("--mode", choices=["stream", "batch", "dashboard", "generate-data", "train"], 
                       default="stream", help="Operation mode")
    parser.add_argument("--data-file", help="Data file for batch mode")
    parser.add_argument("--output-dir", default="data", help="Output directory for generated data")
    parser.add_argument("--no-train", action="store_true", help="Skip model training")
    
    args = parser.parse_args()
    
    # Initialize system
    aml_system = AMLDetectionSystem()
    
    try:
        if args.mode == "generate-data":
            aml_system.generate_test_data(args.output_dir)
            
        elif args.mode == "train":
            aml_system.setup_models(train_models=True)
            
        elif args.mode == "batch":
            if not args.data_file:
                print("Error: --data-file required for batch mode")
                sys.exit(1)
            
            aml_system.setup_models(train_models=not args.no_train)
            results = aml_system.run_batch_analysis(args.data_file)
            
            print(f"\nBatch Analysis Results:")
            print(f"Total anomalies detected: {len(results)}")
            for result in results[:10]:  # Show first 10
                print(f"  Transaction {result['transaction_index']}: "
                      f"{result['anomaly_probability']:.2%} anomaly probability")
                      
        elif args.mode == "dashboard":
            print("Starting dashboard...")
            print("Run: streamlit run src/dashboard/streamlit_app.py")
            
        elif args.mode == "stream":
            aml_system.setup_models(train_models=not args.no_train)
            aml_system.setup_pipeline()
            
            try:
                aml_system.start_streaming()
                print("AML Detection System is running in streaming mode...")
                print("Press Ctrl+C to stop")
                
                # Keep running
                import time
                while True:
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\nShutting down...")
                aml_system.stop_streaming()
                
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()