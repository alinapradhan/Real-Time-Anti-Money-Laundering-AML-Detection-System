from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class TransactionType(str, Enum):
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    PAYMENT = "payment"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Transaction:
    """Financial transaction data model"""
    id: str
    timestamp: datetime
    amount: float
    currency: str = "USD"
    transaction_type: TransactionType = TransactionType.TRANSFER
    source_account: str = ""
    destination_account: Optional[str] = None
    source_country: str = "US"
    destination_country: Optional[str] = None
    merchant_category: Optional[str] = None
    channel: str = "online"
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'amount': self.amount,
            'currency': self.currency,
            'transaction_type': self.transaction_type,
            'source_account': self.source_account,
            'destination_account': self.destination_account,
            'source_country': self.source_country,
            'destination_country': self.destination_country,
            'merchant_category': self.merchant_category,
            'channel': self.channel
        }

@dataclass
class AMLAlert:
    """AML alert data model"""
    id: str
    transaction_id: str
    alert_type: str
    risk_level: RiskLevel
    confidence_score: float
    description: str
    detected_by: str
    timestamp: datetime = field(default_factory=datetime.now)
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'transaction_id': self.transaction_id,
            'alert_type': self.alert_type,
            'risk_level': self.risk_level,
            'confidence_score': self.confidence_score,
            'description': self.description,
            'detected_by': self.detected_by,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'additional_data': self.additional_data
        }

@dataclass
class CustomerProfile:
    """Customer profile for enhanced risk assessment"""
    customer_id: str
    account_age_days: int
    average_monthly_volume: float
    country_residence: str
    customer_type: str = "individual"
    risk_category: RiskLevel = RiskLevel.LOW
    kyc_status: str = "completed"