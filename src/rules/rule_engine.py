import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from collections import defaultdict, deque
from loguru import logger

from ..models.data_models import Transaction, AMLAlert, RiskLevel

class StructuringDetector:
    """Detects structuring - multiple transactions below reporting thresholds"""
    
    def __init__(self, 
                 threshold_amount: float = 10000,
                 time_window_hours: int = 24,
                 min_transactions: int = 3,
                 proximity_percentage: float = 0.9):
        """
        Initialize structuring detector
        
        Args:
            threshold_amount: Amount threshold for structuring detection
            time_window_hours: Time window to look for related transactions
            min_transactions: Minimum number of transactions to flag as structuring
            proximity_percentage: How close to threshold to consider suspicious
        """
        self.threshold_amount = threshold_amount
        self.time_window_hours = time_window_hours
        self.min_transactions = min_transactions
        self.proximity_percentage = proximity_percentage
        self.proximity_threshold = threshold_amount * proximity_percentage
        
        # Track recent transactions by account
        self.account_transactions = defaultdict(lambda: deque(maxlen=100))
    
    def detect(self, transaction: Transaction) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect structuring patterns in a transaction
        
        Args:
            transaction: Transaction to analyze
            
        Returns:
            Tuple of (is_suspicious, additional_data)
        """
        account = transaction.source_account
        current_time = transaction.timestamp
        
        # Add current transaction to history
        self.account_transactions[account].append({
            'timestamp': current_time,
            'amount': transaction.amount,
            'id': transaction.id,
            'type': transaction.transaction_type
        })
        
        # Get transactions within time window
        cutoff_time = current_time - timedelta(hours=self.time_window_hours)
        recent_transactions = [
            tx for tx in self.account_transactions[account]
            if tx['timestamp'] >= cutoff_time
        ]
        
        # Check for structuring patterns
        suspicious_transactions = [
            tx for tx in recent_transactions
            if tx['amount'] >= self.proximity_threshold and tx['amount'] < self.threshold_amount
        ]
        
        if len(suspicious_transactions) >= self.min_transactions:
            total_amount = sum(tx['amount'] for tx in suspicious_transactions)
            
            additional_data = {
                'pattern_type': 'structuring',
                'transaction_count': len(suspicious_transactions),
                'total_amount': total_amount,
                'time_window_hours': self.time_window_hours,
                'threshold_amount': self.threshold_amount,
                'related_transactions': [tx['id'] for tx in suspicious_transactions]
            }
            
            logger.warning(f"Structuring detected: {len(suspicious_transactions)} transactions "
                          f"totaling ${total_amount:.2f} for account {account}")
            
            return True, additional_data
        
        return False, {}

class LayeringDetector:
    """Detects layering - complex transaction patterns to obscure money trail"""
    
    def __init__(self, 
                 complexity_threshold: int = 5,
                 amount_threshold: float = 50000,
                 time_window_hours: int = 72):
        """
        Initialize layering detector
        
        Args:
            complexity_threshold: Minimum number of intermediate accounts/steps
            amount_threshold: Minimum amount to trigger layering detection
            time_window_hours: Time window to track transaction chains
        """
        self.complexity_threshold = complexity_threshold
        self.amount_threshold = amount_threshold
        self.time_window_hours = time_window_hours
        
        # Track transaction chains
        self.transaction_chains = defaultdict(list)
        self.account_connections = defaultdict(set)
    
    def detect(self, transaction: Transaction) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect layering patterns in a transaction
        
        Args:
            transaction: Transaction to analyze
            
        Returns:
            Tuple of (is_suspicious, additional_data)
        """
        if transaction.amount < self.amount_threshold:
            return False, {}
        
        source = transaction.source_account
        destination = transaction.destination_account
        current_time = transaction.timestamp
        
        if not destination:
            return False, {}
        
        # Track account connections
        self.account_connections[source].add(destination)
        
        # Build transaction chain
        chain_key = f"{source}_{destination}_{current_time.date()}"
        self.transaction_chains[chain_key].append({
            'id': transaction.id,
            'timestamp': current_time,
            'amount': transaction.amount,
            'source': source,
            'destination': destination
        })
        
        # Check for complex patterns
        complexity_score = self._calculate_complexity(source, destination, current_time)
        
        if complexity_score >= self.complexity_threshold:
            additional_data = {
                'pattern_type': 'layering',
                'complexity_score': complexity_score,
                'source_connections': len(self.account_connections[source]),
                'destination_account': destination,
                'amount': transaction.amount
            }
            
            logger.warning(f"Layering detected: complexity score {complexity_score} "
                          f"for transaction {transaction.id}")
            
            return True, additional_data
        
        return False, {}
    
    def _calculate_complexity(self, source: str, destination: str, timestamp: datetime) -> int:
        """Calculate complexity score for potential layering"""
        cutoff_time = timestamp - timedelta(hours=self.time_window_hours)
        
        # Count connections within time window
        complexity = 0
        
        # Source account complexity
        complexity += len(self.account_connections[source])
        
        # Look for circular patterns
        if destination in self.account_connections:
            if source in self.account_connections[destination]:
                complexity += 2  # Circular transaction pattern
        
        # Check for rapid succession of transactions
        recent_chains = [
            chain for chain_list in self.transaction_chains.values()
            for chain in chain_list
            if chain['timestamp'] >= cutoff_time and 
            (chain['source'] == source or chain['destination'] == destination)
        ]
        
        if len(recent_chains) > 3:
            complexity += len(recent_chains) // 2
        
        return complexity

class VelocityDetector:
    """Detects unusual transaction velocity patterns"""
    
    def __init__(self, 
                 velocity_threshold: float = 5.0,
                 time_window_minutes: int = 60):
        """
        Initialize velocity detector
        
        Args:
            velocity_threshold: Multiplier for normal velocity to flag as suspicious
            time_window_minutes: Time window for velocity calculation
        """
        self.velocity_threshold = velocity_threshold
        self.time_window_minutes = time_window_minutes
        
        # Track transaction velocities by account
        self.account_velocities = defaultdict(lambda: deque(maxlen=100))
        self.baseline_velocities = defaultdict(float)
    
    def detect(self, transaction: Transaction) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect unusual velocity patterns
        
        Args:
            transaction: Transaction to analyze
            
        Returns:
            Tuple of (is_suspicious, additional_data)
        """
        account = transaction.source_account
        current_time = transaction.timestamp
        
        # Add current transaction
        self.account_velocities[account].append({
            'timestamp': current_time,
            'amount': transaction.amount
        })
        
        # Calculate current velocity
        cutoff_time = current_time - timedelta(minutes=self.time_window_minutes)
        recent_transactions = [
            tx for tx in self.account_velocities[account]
            if tx['timestamp'] >= cutoff_time
        ]
        
        if len(recent_transactions) < 2:
            return False, {}
        
        current_velocity = len(recent_transactions) / (self.time_window_minutes / 60)
        
        # Update baseline velocity (rolling average)
        if account not in self.baseline_velocities:
            self.baseline_velocities[account] = current_velocity
        else:
            self.baseline_velocities[account] = (
                0.9 * self.baseline_velocities[account] + 0.1 * current_velocity
            )
        
        # Check if current velocity is suspicious
        baseline = self.baseline_velocities[account]
        if baseline > 0 and current_velocity > baseline * self.velocity_threshold:
            additional_data = {
                'pattern_type': 'velocity_anomaly',
                'current_velocity': current_velocity,
                'baseline_velocity': baseline,
                'velocity_ratio': current_velocity / baseline,
                'transaction_count': len(recent_transactions),
                'time_window_minutes': self.time_window_minutes
            }
            
            logger.warning(f"Velocity anomaly detected: {current_velocity:.2f} transactions/hour "
                          f"vs baseline {baseline:.2f} for account {account}")
            
            return True, additional_data
        
        return False, {}

class RuleBasedAMLDetector:
    """Main rule-based AML detection engine combining multiple detectors"""
    
    def __init__(self):
        """Initialize the rule-based AML detector"""
        self.structuring_detector = StructuringDetector()
        self.layering_detector = LayeringDetector()
        self.velocity_detector = VelocityDetector()
        
        logger.info("Rule-based AML detector initialized")
    
    def analyze_transaction(self, transaction: Transaction) -> List[AMLAlert]:
        """
        Analyze a transaction using all rule-based detectors
        
        Args:
            transaction: Transaction to analyze
            
        Returns:
            List of AML alerts generated
        """
        alerts = []
        
        # Run all detectors
        detectors = [
            ('structuring', self.structuring_detector),
            ('layering', self.layering_detector),
            ('velocity', self.velocity_detector)
        ]
        
        for detector_name, detector in detectors:
            try:
                is_suspicious, additional_data = detector.detect(transaction)
                
                if is_suspicious:
                    # Determine risk level based on pattern type and severity
                    risk_level = self._determine_risk_level(detector_name, additional_data)
                    
                    alert = AMLAlert(
                        id=f"rule_{detector_name}_{transaction.id}_{datetime.now().timestamp()}",
                        transaction_id=transaction.id,
                        alert_type=f"rule_based_{detector_name}",
                        risk_level=risk_level,
                        confidence_score=0.8,  # Rule-based alerts have high confidence
                        description=self._generate_alert_description(detector_name, additional_data),
                        detected_by=f"rule_engine_{detector_name}",
                        additional_data=additional_data
                    )
                    
                    alerts.append(alert)
                    
            except Exception as e:
                logger.error(f"Error in {detector_name} detector: {str(e)}")
        
        return alerts
    
    def _determine_risk_level(self, detector_name: str, additional_data: Dict[str, Any]) -> RiskLevel:
        """Determine risk level based on detection type and severity"""
        if detector_name == 'structuring':
            if additional_data.get('total_amount', 0) > 100000:
                return RiskLevel.HIGH
            elif additional_data.get('transaction_count', 0) > 5:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        elif detector_name == 'layering':
            if additional_data.get('complexity_score', 0) > 10:
                return RiskLevel.CRITICAL
            elif additional_data.get('complexity_score', 0) > 7:
                return RiskLevel.HIGH
            else:
                return RiskLevel.MEDIUM
                
        elif detector_name == 'velocity':
            if additional_data.get('velocity_ratio', 0) > 10:
                return RiskLevel.HIGH
            elif additional_data.get('velocity_ratio', 0) > 5:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
        
        return RiskLevel.LOW
    
    def _generate_alert_description(self, detector_name: str, additional_data: Dict[str, Any]) -> str:
        """Generate human-readable alert descriptions"""
        if detector_name == 'structuring':
            return (f"Structuring detected: {additional_data.get('transaction_count', 0)} "
                   f"transactions totaling ${additional_data.get('total_amount', 0):,.2f} "
                   f"in {additional_data.get('time_window_hours', 0)} hours")
                   
        elif detector_name == 'layering':
            return (f"Layering detected: complexity score {additional_data.get('complexity_score', 0)} "
                   f"for ${additional_data.get('amount', 0):,.2f} transaction")
                   
        elif detector_name == 'velocity':
            return (f"Velocity anomaly: {additional_data.get('current_velocity', 0):.1f} "
                   f"transactions/hour vs baseline {additional_data.get('baseline_velocity', 0):.1f}")
        
        return f"Rule-based detection: {detector_name}"