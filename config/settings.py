# Configuration settings for AML Detection System

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TRANSACTION_TOPIC = "financial-transactions"
KAFKA_ALERT_TOPIC = "aml-alerts"

# ML Model Configuration
ISOLATION_FOREST_CONTAMINATION = 0.1
AUTOENCODER_THRESHOLD = 0.95

# Rule-based Detection Thresholds
STRUCTURING_THRESHOLD = 10000  # Amount in USD
STRUCTURING_TIME_WINDOW = 24   # Hours
LAYERING_COMPLEXITY_THRESHOLD = 5  # Number of intermediate accounts

# Alert Configuration
ALERT_CHANNELS = ["email", "dashboard", "log"]
HIGH_RISK_THRESHOLD = 0.8
MEDIUM_RISK_THRESHOLD = 0.5

# Dashboard Configuration
DASHBOARD_REFRESH_INTERVAL = 5  # Seconds
MAX_TRANSACTIONS_DISPLAY = 1000

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "logs/aml_system.log"