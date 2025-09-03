import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import random
from typing import List, Dict, Any
import json

from ..models.data_models import Transaction, TransactionType, CustomerProfile, RiskLevel

class TransactionGenerator:
    """Generate synthetic financial transaction data for testing"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize transaction generator
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Define realistic data distributions
        self.account_ids = [f"ACC{i:06d}" for i in range(1000, 10000)]
        self.countries = ["US", "UK", "CA", "DE", "FR", "JP", "AU", "SG", "HK", "CH"]
        self.channels = ["online", "atm", "branch", "mobile"]
        self.merchant_categories = ["grocery", "gas", "restaurant", "retail", "pharmacy", "entertainment"]
        
        # AML patterns for suspicious transactions
        self.structuring_accounts = random.sample(self.account_ids, 20)
        self.layering_networks = self._create_layering_networks()
    
    def _create_layering_networks(self) -> Dict[str, List[str]]:
        """Create networks of accounts for layering patterns"""
        networks = {}
        for i in range(5):
            hub_account = random.choice(self.account_ids)
            connected_accounts = random.sample(
                [acc for acc in self.account_ids if acc != hub_account], 
                random.randint(5, 15)
            )
            networks[hub_account] = connected_accounts
        return networks
    
    def generate_normal_transactions(self, n_transactions: int, 
                                   start_date: datetime = None) -> List[Transaction]:
        """
        Generate normal (non-suspicious) transactions
        
        Args:
            n_transactions: Number of transactions to generate
            start_date: Start date for transactions
            
        Returns:
            List of Transaction objects
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        
        transactions = []
        
        for i in range(n_transactions):
            # Generate realistic transaction
            transaction = self._generate_normal_transaction(start_date, i)
            transactions.append(transaction)
        
        return transactions
    
    def generate_suspicious_transactions(self, n_transactions: int = 100,
                                       pattern_type: str = "mixed") -> List[Transaction]:
        """
        Generate suspicious transactions with AML patterns
        
        Args:
            n_transactions: Number of suspicious transactions
            pattern_type: Type of pattern ('structuring', 'layering', 'velocity', 'mixed')
            
        Returns:
            List of suspicious Transaction objects
        """
        transactions = []
        start_date = datetime.now() - timedelta(days=7)
        
        if pattern_type in ["structuring", "mixed"]:
            transactions.extend(self._generate_structuring_pattern(n_transactions // 3, start_date))
        
        if pattern_type in ["layering", "mixed"]:
            transactions.extend(self._generate_layering_pattern(n_transactions // 3, start_date))
        
        if pattern_type in ["velocity", "mixed"]:
            transactions.extend(self._generate_velocity_pattern(n_transactions // 3, start_date))
        
        return transactions
    
    def _generate_normal_transaction(self, base_date: datetime, index: int) -> Transaction:
        """Generate a single normal transaction"""
        # Time distribution (more transactions during business hours)
        hours_offset = np.random.exponential(scale=12) % 24
        if random.random() < 0.7:  # 70% during business hours
            hours_offset = random.uniform(8, 18)
        
        timestamp = base_date + timedelta(
            days=random.uniform(0, 30),
            hours=hours_offset,
            minutes=random.uniform(0, 59)
        )
        
        # Amount distribution (log-normal for realistic distribution)
        amount = np.random.lognormal(mean=6, sigma=1.2)
        amount = round(max(1, min(amount, 50000)), 2)  # Cap at $50k
        
        # Transaction type based on amount
        if amount < 100:
            txn_type = random.choice([TransactionType.PAYMENT, TransactionType.WITHDRAWAL])
        elif amount < 5000:
            txn_type = random.choice([TransactionType.PAYMENT, TransactionType.TRANSFER, TransactionType.DEPOSIT])
        else:
            txn_type = random.choice([TransactionType.TRANSFER, TransactionType.DEPOSIT])
        
        source_account = random.choice(self.account_ids)
        destination_account = None
        if txn_type in [TransactionType.TRANSFER, TransactionType.PAYMENT]:
            destination_account = random.choice([acc for acc in self.account_ids if acc != source_account])
        
        return Transaction(
            id=f"TXN{timestamp.strftime('%Y%m%d')}{index:06d}",
            timestamp=timestamp,
            amount=amount,
            currency="USD",
            transaction_type=txn_type,
            source_account=source_account,
            destination_account=destination_account,
            source_country=random.choice(self.countries),
            destination_country=random.choice(self.countries) if destination_account else None,
            merchant_category=random.choice(self.merchant_categories) if txn_type == TransactionType.PAYMENT else None,
            channel=random.choice(self.channels)
        )
    
    def _generate_structuring_pattern(self, n_transactions: int, base_date: datetime) -> List[Transaction]:
        """Generate structuring pattern (multiple transactions just below threshold)"""
        transactions = []
        
        for account in random.sample(self.structuring_accounts, min(5, len(self.structuring_accounts))):
            # Generate 3-8 transactions per account
            n_txns = random.randint(3, 8)
            base_time = base_date + timedelta(hours=random.uniform(0, 48))
            
            for i in range(n_txns):
                # Amounts just below $10,000 threshold
                amount = random.uniform(9000, 9950)
                
                timestamp = base_time + timedelta(minutes=random.uniform(30, 240))
                
                transaction = Transaction(
                    id=f"STRUCT{timestamp.strftime('%Y%m%d%H%M')}{i:03d}",
                    timestamp=timestamp,
                    amount=round(amount, 2),
                    currency="USD",
                    transaction_type=TransactionType.WITHDRAWAL,
                    source_account=account,
                    destination_account=None,
                    source_country="US",
                    destination_country=None,
                    channel="atm"
                )
                
                transactions.append(transaction)
                
                if len(transactions) >= n_transactions:
                    break
            
            if len(transactions) >= n_transactions:
                break
        
        return transactions
    
    def _generate_layering_pattern(self, n_transactions: int, base_date: datetime) -> List[Transaction]:
        """Generate layering pattern (complex multi-hop transfers)"""
        transactions = []
        
        for hub_account, connected_accounts in list(self.layering_networks.items())[:3]:
            # Large initial amount
            initial_amount = random.uniform(100000, 500000)
            current_amount = initial_amount
            current_account = hub_account
            
            # Create chain of transfers
            for i in range(min(8, len(connected_accounts))):
                next_account = connected_accounts[i]
                transfer_amount = current_amount * random.uniform(0.8, 0.95)  # Keep most of the money
                
                timestamp = base_date + timedelta(
                    hours=random.uniform(i * 2, i * 2 + 4)  # Space out transfers
                )
                
                transaction = Transaction(
                    id=f"LAYER{timestamp.strftime('%Y%m%d%H%M')}{i:03d}",
                    timestamp=timestamp,
                    amount=round(transfer_amount, 2),
                    currency="USD",
                    transaction_type=TransactionType.TRANSFER,
                    source_account=current_account,
                    destination_account=next_account,
                    source_country=random.choice(self.countries),
                    destination_country=random.choice(self.countries),
                    channel="online"
                )
                
                transactions.append(transaction)
                current_account = next_account
                current_amount = transfer_amount
                
                if len(transactions) >= n_transactions:
                    break
            
            if len(transactions) >= n_transactions:
                break
        
        return transactions
    
    def _generate_velocity_pattern(self, n_transactions: int, base_date: datetime) -> List[Transaction]:
        """Generate high-velocity transaction pattern"""
        transactions = []
        
        # Select a few accounts for high velocity
        velocity_accounts = random.sample(self.account_ids, 3)
        
        for account in velocity_accounts:
            # Generate many transactions in short time period
            n_txns = random.randint(15, 25)
            base_time = base_date + timedelta(hours=random.uniform(0, 24))
            
            for i in range(n_txns):
                # Small to medium amounts
                amount = random.uniform(500, 5000)
                
                # Very close together in time
                timestamp = base_time + timedelta(minutes=random.uniform(0, 120))
                
                transaction = Transaction(
                    id=f"VELOCITY{timestamp.strftime('%Y%m%d%H%M')}{i:03d}",
                    timestamp=timestamp,
                    amount=round(amount, 2),
                    currency="USD",
                    transaction_type=random.choice([TransactionType.TRANSFER, TransactionType.PAYMENT]),
                    source_account=account,
                    destination_account=random.choice([acc for acc in self.account_ids if acc != account]),
                    source_country="US",
                    destination_country=random.choice(self.countries),
                    channel=random.choice(["online", "mobile"])
                )
                
                transactions.append(transaction)
                
                if len(transactions) >= n_transactions:
                    break
            
            if len(transactions) >= n_transactions:
                break
        
        return transactions
    
    def generate_customer_profiles(self, n_customers: int = 1000) -> List[CustomerProfile]:
        """Generate customer profiles"""
        profiles = []
        
        for i in range(n_customers):
            profile = CustomerProfile(
                customer_id=f"CUST{i:06d}",
                account_age_days=random.randint(30, 3650),  # 1 month to 10 years
                average_monthly_volume=np.random.lognormal(mean=8, sigma=1.5),
                country_residence=random.choice(self.countries),
                customer_type=random.choice(["individual", "business"]),
                risk_category=random.choice(list(RiskLevel)),
                kyc_status=random.choice(["completed", "pending", "expired"])
            )
            profiles.append(profile)
        
        return profiles
    
    def save_transactions_to_csv(self, transactions: List[Transaction], filepath: str):
        """Save transactions to CSV file"""
        df = pd.DataFrame([txn.dict() for txn in transactions])
        df.to_csv(filepath, index=False)
        print(f"Saved {len(transactions)} transactions to {filepath}")
    
    def save_transactions_to_json(self, transactions: List[Transaction], filepath: str):
        """Save transactions to JSON file"""
        data = [txn.dict() for txn in transactions]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved {len(transactions)} transactions to {filepath}")

def generate_sample_data():
    """Generate sample data for testing the AML system"""
    generator = TransactionGenerator()
    
    print("Generating sample transaction data...")
    
    # Generate normal transactions
    normal_transactions = generator.generate_normal_transactions(5000)
    
    # Generate suspicious transactions
    suspicious_transactions = generator.generate_suspicious_transactions(200, "mixed")
    
    # Combine all transactions
    all_transactions = normal_transactions + suspicious_transactions
    
    # Shuffle to mix normal and suspicious
    random.shuffle(all_transactions)
    
    # Sort by timestamp
    all_transactions.sort(key=lambda x: x.timestamp)
    
    return all_transactions