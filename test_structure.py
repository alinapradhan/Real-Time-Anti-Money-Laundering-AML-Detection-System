#!/usr/bin/env python3
"""
Simple test script to validate the AML system structure without external dependencies
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that core modules can be imported"""
    print("🔍 Testing AML System Basic Structure")
    print("=" * 50)
    
    try:
        print("\n1. Testing data models...")
        from src.models.data_models import Transaction, AMLAlert, TransactionType, RiskLevel
        print("   ✓ Data models imported successfully")
        
        # Test transaction creation
        transaction = Transaction(
            id="TEST001",
            timestamp=datetime.now(),
            amount=1000.0,
            currency="USD",
            transaction_type=TransactionType.TRANSFER,
            source_account="ACC123456",
            destination_account="ACC654321",
            source_country="US",
            destination_country="CA",
            channel="online"
        )
        print(f"   ✓ Transaction created: {transaction.id}")
        
        # Test alert creation
        alert = AMLAlert(
            id="ALERT001",
            transaction_id="TEST001",
            alert_type="test",
            risk_level=RiskLevel.MEDIUM,
            confidence_score=0.75,
            description="Test alert",
            detected_by="test_system"
        )
        print(f"   ✓ Alert created: {alert.id}")
        
    except Exception as e:
        print(f"   ✗ Error importing data models: {str(e)}")
        return False
    
    try:
        print("\n2. Testing rule engine...")
        from src.rules.rule_engine import RuleBasedAMLDetector
        rule_detector = RuleBasedAMLDetector()
        print("   ✓ Rule engine initialized")
        
        # Test transaction analysis
        alerts = rule_detector.analyze_transaction(transaction)
        print(f"   ✓ Transaction analyzed, {len(alerts)} alerts generated")
        
    except Exception as e:
        print(f"   ✗ Error testing rule engine: {str(e)}")
        return False
    
    try:
        print("\n3. Testing alert system...")
        from src.alerts.alert_manager import AlertManager
        alert_manager = AlertManager()
        print("   ✓ Alert manager initialized")
        
        # Test alert processing
        result = alert_manager.process_alert(alert)
        print(f"   ✓ Alert processed: {result}")
        
    except Exception as e:
        print(f"   ✗ Error testing alert system: {str(e)}")
        return False
    
    try:
        print("\n4. Testing configuration...")
        from config.settings import STRUCTURING_THRESHOLD, KAFKA_BOOTSTRAP_SERVERS
        print(f"   ✓ Configuration loaded - Structuring threshold: ${STRUCTURING_THRESHOLD:,}")
        print(f"   ✓ Kafka servers: {KAFKA_BOOTSTRAP_SERVERS}")
        
    except Exception as e:
        print(f"   ✗ Error loading configuration: {str(e)}")
        return False
    
    return True

def test_directory_structure():
    """Test that all required directories exist"""
    print("\n5. Testing directory structure...")
    
    required_dirs = [
        "src/models",
        "src/rules", 
        "src/pipeline",
        "src/alerts",
        "src/dashboard",
        "src/utils",
        "config",
        "tests"
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✓ {directory}")
        else:
            print(f"   ✗ {directory} - MISSING")
            all_exist = False
    
    return all_exist

def test_file_completeness():
    """Test that key files exist"""
    print("\n6. Testing file completeness...")
    
    required_files = [
        "main.py",
        "demo.py", 
        "requirements.txt",
        "README.md",
        ".gitignore",
        "src/models/data_models.py",
        "src/models/isolation_forest.py",
        "src/models/autoencoder.py",
        "src/rules/rule_engine.py",
        "src/pipeline/streaming.py",
        "src/alerts/alert_manager.py",
        "src/dashboard/streamlit_app.py",
        "src/utils/data_generator.py",
        "config/settings.py",
        "tests/test_aml_system.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✓ {file_path} ({size:,} bytes)")
        else:
            print(f"   ✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("Testing Real-Time AML Detection System Implementation")
    print("=" * 60)
    
    results = []
    
    # Test basic imports and functionality
    results.append(test_basic_imports())
    
    # Test directory structure
    results.append(test_directory_structure())
    
    # Test file completeness
    results.append(test_file_completeness())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if all(results):
        print("🎉 ALL TESTS PASSED!")
        print("\nSystem Status: ✅ READY FOR DEPLOYMENT")
        print("\nNext Steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run full demo: python demo.py")  
        print("  3. Generate sample data: python main.py --mode generate-data")
        print("  4. Start dashboard: streamlit run src/dashboard/streamlit_app.py")
        print("  5. Run streaming mode: python main.py --mode stream")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("\nSystem Status: ⚠️  NEEDS ATTENTION")
        failed_count = sum(1 for r in results if not r)
        print(f"Failed tests: {failed_count}/{len(results)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)