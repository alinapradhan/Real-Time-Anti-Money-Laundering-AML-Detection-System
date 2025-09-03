from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
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

class Transaction(BaseModel):
    """Financial transaction data model"""
    id: str = Field(..., description="Unique transaction identifier")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(default="USD", description="Currency code")
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    source_account: str = Field(..., description="Source account identifier")
    destination_account: Optional[str] = Field(None, description="Destination account identifier")
    source_country: str = Field(..., description="Source country code")
    destination_country: Optional[str] = Field(None, description="Destination country code")
    merchant_category: Optional[str] = Field(None, description="Merchant category code")
    channel: str = Field(..., description="Transaction channel (online, atm, branch)")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AMLAlert(BaseModel):
    """AML alert data model"""
    id: str = Field(..., description="Unique alert identifier")
    transaction_id: str = Field(..., description="Associated transaction ID")
    alert_type: str = Field(..., description="Type of AML violation detected")
    risk_level: RiskLevel = Field(..., description="Risk assessment level")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence score")
    description: str = Field(..., description="Alert description")
    detected_by: str = Field(..., description="Detection method (model/rule)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Alert timestamp")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class CustomerProfile(BaseModel):
    """Customer profile for enhanced risk assessment"""
    customer_id: str = Field(..., description="Unique customer identifier")
    account_age_days: int = Field(..., ge=0, description="Account age in days")
    average_monthly_volume: float = Field(..., ge=0, description="Average monthly transaction volume")
    country_residence: str = Field(..., description="Country of residence")
    customer_type: str = Field(..., description="Customer type (individual/business)")
    risk_category: RiskLevel = Field(default=RiskLevel.LOW, description="Customer risk category")
    kyc_status: str = Field(default="completed", description="KYC completion status")