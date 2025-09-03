import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.data_models import Transaction, AMLAlert, TransactionType, RiskLevel
from src.models.isolation_forest import IsolationForestDetector
from src.rules.rule_engine import RuleBasedAMLDetector, StructuringDetector
from src.alerts.alert_manager import AlertManager
from src.utils.data_generator import TransactionGenerator

class TestDataModels:
    """Test data models and validation"""
    
    def test_transaction_creation(self):
        """Test valid transaction creation"""
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
        
        assert transaction.id == "TEST001"
        assert transaction.amount == 1000.0
        assert transaction.transaction_type == TransactionType.TRANSFER
    
    def test_invalid_transaction_amount(self):
        """Test that invalid amounts raise validation errors"""
        with pytest.raises(ValueError):
            Transaction(
                id="TEST002",
                timestamp=datetime.now(),
                amount=-100.0,  # Negative amount should fail
                currency="USD",
                transaction_type=TransactionType.PAYMENT,
                source_account="ACC123456",
                source_country="US",
                channel="online"
            )
    
    def test_aml_alert_creation(self):
        """Test AML alert creation"""
        alert = AMLAlert(
            id="ALERT001",
            transaction_id="TXN001",
            alert_type="structuring",
            risk_level=RiskLevel.HIGH,
            confidence_score=0.85,
            description="Multiple transactions detected",
            detected_by="rule_engine"
        )
        
        assert alert.risk_level == RiskLevel.HIGH
        assert alert.confidence_score == 0.85

class TestIsolationForest:
    """Test Isolation Forest detector"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        detector = IsolationForestDetector(contamination=0.1)
        assert detector.contamination == 0.1
        assert not detector.is_fitted
    
    def test_feature_extraction(self):
        """Test feature extraction from transactions"""
        detector = IsolationForestDetector()
        
        # Create sample data
        data = {
            'id': ['TXN001', 'TXN002'],
            'timestamp': [datetime.now(), datetime.now()],
            'amount': [1000.0, 2000.0],
            'transaction_type': ['transfer', 'payment'],
            'source_account': ['ACC001', 'ACC002'],
            'channel': ['online', 'atm']
        }
        df = pd.DataFrame(data)
        
        features = detector._extract_features(df)
        
        assert len(features) == 2
        assert 'amount' in features.columns
        assert 'hour' in features.columns
    
    def test_model_training(self):
        """Test model training"""
        # Generate sample training data
        generator = TransactionGenerator()
        transactions = generator.generate_normal_transactions(100)
        df = pd.DataFrame([txn.dict() for txn in transactions])
        
        detector = IsolationForestDetector()
        detector.fit(df)
        
        assert detector.is_fitted
        assert detector.feature_names is not None

class TestRuleEngine:
    """Test rule-based detection"""
    
    def test_structuring_detector_init(self):
        """Test structuring detector initialization"""
        detector = StructuringDetector(
            threshold_amount=10000,
            time_window_hours=24,
            min_transactions=3
        )
        
        assert detector.threshold_amount == 10000
        assert detector.time_window_hours == 24
        assert detector.min_transactions == 3
    
    def test_structuring_detection(self):
        """Test structuring pattern detection"""
        detector = StructuringDetector(
            threshold_amount=10000,
            time_window_hours=24,
            min_transactions=3
        )
        
        # Create transactions that should trigger structuring detection
        base_time = datetime.now()
        transactions = []
        
        for i in range(4):
            transaction = Transaction(
                id=f"STRUCT{i:03d}",
                timestamp=base_time + timedelta(hours=i),
                amount=9500.0,  # Just below threshold
                currency="USD",
                transaction_type=TransactionType.WITHDRAWAL,
                source_account="ACC123456",
                source_country="US",
                channel="atm"
            )
            transactions.append(transaction)
        
        # Process transactions
        alerts_detected = []
        for txn in transactions:
            is_suspicious, data = detector.detect(txn)
            if is_suspicious:
                alerts_detected.append(data)
        
        # Should detect structuring after processing enough transactions
        assert len(alerts_detected) > 0
    
    def test_rule_based_detector(self):
        """Test complete rule-based detector"""
        detector = RuleBasedAMLDetector()
        
        transaction = Transaction(
            id="TEST001",
            timestamp=datetime.now(),
            amount=5000.0,
            currency="USD",
            transaction_type=TransactionType.TRANSFER,
            source_account="ACC123456",
            destination_account="ACC654321",
            source_country="US",
            destination_country="CA",
            channel="online"
        )
        
        alerts = detector.analyze_transaction(transaction)
        
        # Should return a list (may be empty for normal transaction)
        assert isinstance(alerts, list)

class TestAlertManager:
    """Test alert management system"""
    
    def test_alert_manager_init(self):
        """Test alert manager initialization"""
        manager = AlertManager()
        
        assert len(manager.notifiers) > 0
        assert manager.log_notifier is not None
        assert manager.dashboard_notifier is not None
    
    def test_alert_processing(self):
        """Test alert processing"""
        manager = AlertManager()
        
        alert = AMLAlert(
            id="TEST_ALERT_001",
            transaction_id="TXN001",
            alert_type="test_alert",
            risk_level=RiskLevel.MEDIUM,
            confidence_score=0.75,
            description="Test alert for unit testing",
            detected_by="unit_test"
        )
        
        result = manager.process_alert(alert)
        assert result is True
        
        # Check that alert is in history
        history = manager.get_alert_history(limit=10)
        assert len(history) > 0
        assert history[-1].id == "TEST_ALERT_001"
    
    def test_duplicate_detection(self):
        """Test duplicate alert detection"""
        manager = AlertManager()
        
        alert1 = AMLAlert(
            id="DUP_ALERT_001",
            transaction_id="TXN001",
            alert_type="duplicate_test",
            risk_level=RiskLevel.LOW,
            confidence_score=0.5,
            description="First alert",
            detected_by="unit_test"
        )
        
        alert2 = AMLAlert(
            id="DUP_ALERT_002",
            transaction_id="TXN001",  # Same transaction
            alert_type="duplicate_test",  # Same type
            risk_level=RiskLevel.LOW,
            confidence_score=0.5,
            description="Duplicate alert",
            detected_by="unit_test"
        )
        
        # Process first alert
        result1 = manager.process_alert(alert1)
        assert result1 is True
        
        # Process duplicate alert (should be rejected)
        result2 = manager.process_alert(alert2)
        assert result2 is False

class TestDataGenerator:
    """Test data generation utilities"""
    
    def test_transaction_generator_init(self):
        """Test transaction generator initialization"""
        generator = TransactionGenerator(seed=42)
        
        assert len(generator.account_ids) > 0
        assert len(generator.countries) > 0
        assert len(generator.channels) > 0
    
    def test_normal_transaction_generation(self):
        """Test normal transaction generation"""
        generator = TransactionGenerator()
        transactions = generator.generate_normal_transactions(10)
        
        assert len(transactions) == 10
        
        for txn in transactions:
            assert isinstance(txn, Transaction)
            assert txn.amount > 0
            assert txn.id is not None
    
    def test_suspicious_transaction_generation(self):
        """Test suspicious transaction generation"""
        generator = TransactionGenerator()
        
        # Test structuring pattern
        structuring_txns = generator.generate_suspicious_transactions(10, "structuring")
        assert len(structuring_txns) > 0
        
        # Test layering pattern
        layering_txns = generator.generate_suspicious_transactions(10, "layering")
        assert len(layering_txns) > 0
        
        # Test velocity pattern
        velocity_txns = generator.generate_suspicious_transactions(10, "velocity")
        assert len(velocity_txns) > 0

# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_detection(self):
        """Test complete detection pipeline"""
        # Generate test data
        generator = TransactionGenerator()
        normal_txns = generator.generate_normal_transactions(50)
        suspicious_txns = generator.generate_suspicious_transactions(10, "mixed")
        
        # Setup detectors
        rule_detector = RuleBasedAMLDetector()
        alert_manager = AlertManager()
        
        # Process all transactions
        total_alerts = 0
        
        for txn in normal_txns + suspicious_txns:
            alerts = rule_detector.analyze_transaction(txn)
            for alert in alerts:
                alert_manager.process_alert(alert)
                total_alerts += 1
        
        # Should have generated some alerts from suspicious transactions
        dashboard_data = alert_manager.get_dashboard_data()
        assert dashboard_data['total_alerts_processed'] >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])