#!/usr/bin/env python3
"""
Minimal demo script using only standard library
"""

import os
import sys
import json
from datetime import datetime, timedelta
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_minimal_demo():
    """Run a minimal demonstration using only standard library"""
    print("🔍 Real-Time AML Detection System - Minimal Demo")
    print("=" * 55)
    
    # Import simplified models
    try:
        print("\n1. Loading system components...")
        from src.models.simple_models import Transaction, AMLAlert, TransactionType, RiskLevel
        print("   ✓ Data models loaded")
    except Exception as e:
        print(f"   ✗ Error loading models: {e}")
        return
    
    # Create sample transactions
    print("\n2. Creating sample transactions...")
    transactions = []
    
    # Normal transactions
    for i in range(5):
        transaction = Transaction(
            id=f"TXN{i:03d}",
            timestamp=datetime.now() - timedelta(hours=i),
            amount=round(random.uniform(100, 5000), 2),
            transaction_type=random.choice(list(TransactionType)),
            source_account=f"ACC{random.randint(1000, 9999)}",
            destination_account=f"ACC{random.randint(1000, 9999)}",
            source_country="US",
            channel=random.choice(["online", "atm", "branch"])
        )
        transactions.append(transaction)
    
    # Suspicious transactions (structuring pattern)
    print("   Creating suspicious structuring pattern...")
    base_time = datetime.now()
    suspicious_account = "ACC9999"
    
    for i in range(4):
        transaction = Transaction(
            id=f"SUSPICIOUS{i:03d}",
            timestamp=base_time + timedelta(hours=i),
            amount=9500.0,  # Just below $10k threshold
            transaction_type=TransactionType.WITHDRAWAL,
            source_account=suspicious_account,
            source_country="US",
            channel="atm"
        )
        transactions.append(transaction)
    
    print(f"   ✓ Created {len(transactions)} transactions")
    
    # Analyze transactions using simple rules
    print("\n3. Analyzing transactions with rule-based detection...")
    alerts = []
    
    # Simple structuring detection
    account_amounts = {}
    for txn in transactions:
        if txn.source_account not in account_amounts:
            account_amounts[txn.source_account] = []
        account_amounts[txn.source_account].append(txn.amount)
    
    for account, amounts in account_amounts.items():
        # Check for structuring pattern
        large_amounts = [amt for amt in amounts if amt >= 9000 and amt < 10000]
        if len(large_amounts) >= 3:
            alert = AMLAlert(
                id=f"STRUCT_ALERT_{account}",
                transaction_id="MULTIPLE",
                alert_type="structuring",
                risk_level=RiskLevel.HIGH,
                confidence_score=0.85,
                description=f"Structuring detected: {len(large_amounts)} transactions near $10k threshold",
                detected_by="simple_rule_engine",
                additional_data={
                    "account": account,
                    "transaction_count": len(large_amounts),
                    "total_amount": sum(large_amounts)
                }
            )
            alerts.append(alert)
    
    # Simple velocity detection
    for account, amounts in account_amounts.items():
        if len(amounts) > 3:  # More than 3 transactions per account
            alert = AMLAlert(
                id=f"VELOCITY_ALERT_{account}",
                transaction_id="MULTIPLE", 
                alert_type="high_velocity",
                risk_level=RiskLevel.MEDIUM,
                confidence_score=0.70,
                description=f"High velocity detected: {len(amounts)} transactions",
                detected_by="simple_rule_engine",
                additional_data={
                    "account": account,
                    "transaction_count": len(amounts)
                }
            )
            alerts.append(alert)
    
    print(f"   ✓ Analysis complete - {len(alerts)} alerts generated")
    
    # Display results
    print("\n4. Detection Results:")
    print(f"   Total transactions processed: {len(transactions)}")
    print(f"   Suspicious patterns detected: {len(alerts)}")
    
    if alerts:
        print("\n5. Alert Details:")
        for i, alert in enumerate(alerts, 1):
            print(f"\n   Alert #{i}:")
            print(f"     ID: {alert.id}")
            print(f"     Type: {alert.alert_type}")
            print(f"     Risk Level: {alert.risk_level.upper()}")
            print(f"     Confidence: {alert.confidence_score:.1%}")
            print(f"     Description: {alert.description}")
            
            if alert.additional_data:
                print(f"     Additional Info:")
                for key, value in alert.additional_data.items():
                    print(f"       {key}: {value}")
    
    # Show transaction summary
    print("\n6. Transaction Summary:")
    print("   Account Activity:")
    for account, amounts in account_amounts.items():
        total = sum(amounts)
        avg = total / len(amounts)
        print(f"     {account}: {len(amounts)} txns, ${total:,.2f} total, ${avg:,.2f} avg")
    
    # Generate simple report
    print("\n7. Generating JSON report...")
    report = {
        "analysis_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_transactions": len(transactions),
            "total_alerts": len(alerts),
            "accounts_analyzed": len(account_amounts)
        },
        "alerts": [alert.dict() for alert in alerts],
        "account_summary": {
            account: {
                "transaction_count": len(amounts),
                "total_amount": sum(amounts),
                "average_amount": sum(amounts) / len(amounts)
            }
            for account, amounts in account_amounts.items()
        }
    }
    
    # Save report
    with open("aml_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("   ✓ Report saved to 'aml_analysis_report.json'")
    
    print("\n🎉 Minimal demo completed successfully!")
    print("\nSystem Capabilities Demonstrated:")
    print("  ✓ Transaction data modeling")
    print("  ✓ Structuring detection (multiple transactions below threshold)")
    print("  ✓ Velocity analysis (high transaction frequency)")
    print("  ✓ Risk level assessment")
    print("  ✓ Alert generation and reporting")
    print("  ✓ JSON report generation")
    
    print(f"\nTo run the full system with ML capabilities:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run full demo: python demo.py")
    print("  3. Launch dashboard: streamlit run src/dashboard/streamlit_app.py")

if __name__ == "__main__":
    run_minimal_demo()