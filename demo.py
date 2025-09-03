#!/usr/bin/env python3
"""
Quick demo script to showcase the AML detection system
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.isolation_forest import IsolationForestDetector
from src.rules.rule_engine import RuleBasedAMLDetector
from src.alerts.alert_manager import AlertManager
from src.utils.data_generator import TransactionGenerator
from src.models.data_models import Transaction

def run_demo():
    """Run a quick demonstration of the AML system"""
    print("🔍 Real-Time AML Detection System Demo")
    print("=" * 50)
    
    # Initialize components
    print("\n1. Initializing system components...")
    generator = TransactionGenerator()
    isolation_forest = IsolationForestDetector()
    rule_detector = RuleBasedAMLDetector()
    alert_manager = AlertManager()
    
    # Generate training data
    print("\n2. Generating training data...")
    training_transactions = generator.generate_normal_transactions(1000)
    training_df = pd.DataFrame([txn.dict() for txn in training_transactions])
    print(f"Generated {len(training_transactions)} normal transactions for training")
    
    # Train ML model
    print("\n3. Training Isolation Forest model...")
    isolation_forest.fit(training_df)
    print("✓ Model trained successfully")
    
    # Generate test data with suspicious patterns
    print("\n4. Generating test data with suspicious patterns...")
    test_transactions = (
        generator.generate_normal_transactions(50) +
        generator.generate_suspicious_transactions(20, "mixed")
    )
    print(f"Generated {len(test_transactions)} test transactions")
    
    # Analyze transactions
    print("\n5. Analyzing transactions...")
    total_alerts = 0
    
    for i, transaction in enumerate(test_transactions):
        print(f"\rProcessing transaction {i+1}/{len(test_transactions)}", end="", flush=True)
        
        # Rule-based detection
        rule_alerts = rule_detector.analyze_transaction(transaction)
        
        # ML-based detection
        ml_alerts = []
        try:
            df_single = pd.DataFrame([transaction.dict()])
            predictions, scores = isolation_forest.predict(df_single)
            
            if predictions[0] == -1:  # Anomaly detected
                from src.models.data_models import AMLAlert
                ml_alert = AMLAlert(
                    id=f"ml_demo_{transaction.id}",
                    transaction_id=transaction.id,
                    alert_type="ml_anomaly",
                    risk_level="medium",
                    confidence_score=0.8,
                    description="ML model detected anomaly",
                    detected_by="isolation_forest"
                )
                ml_alerts.append(ml_alert)
        except:
            pass
        
        # Process all alerts
        all_alerts = rule_alerts + ml_alerts
        for alert in all_alerts:
            alert_manager.process_alert(alert)
            total_alerts += 1
    
    print(f"\n✓ Analysis complete")
    
    # Show results
    print("\n6. Detection Results:")
    print(f"   Total transactions processed: {len(test_transactions)}")
    print(f"   Total alerts generated: {total_alerts}")
    
    # Get alert statistics
    dashboard_data = alert_manager.get_dashboard_data()
    stats = dashboard_data['stats']
    
    print("\n   Alert breakdown:")
    for risk_level in ['critical', 'high', 'medium', 'low']:
        count = stats.get(risk_level, 0)
        if count > 0:
            print(f"     {risk_level.capitalize()}: {count}")
    
    # Show recent alerts
    recent_alerts = alert_manager.get_alert_history(limit=5)
    if recent_alerts:
        print("\n7. Recent Alerts (Top 5):")
        for alert in recent_alerts:
            print(f"   • {alert.alert_type} - {alert.risk_level.upper()}")
            print(f"     Transaction: {alert.transaction_id}")
            print(f"     Confidence: {alert.confidence_score:.2%}")
            print(f"     Description: {alert.description}")
            print()
    
    print("🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("  • Run 'python main.py --mode generate-data' to create larger datasets")
    print("  • Run 'python main.py --mode train' to train models on larger data")
    print("  • Run 'streamlit run src/dashboard/streamlit_app.py' for the dashboard")
    print("  • Run 'python main.py --mode stream' for real-time detection")

if __name__ == "__main__":
    run_demo()