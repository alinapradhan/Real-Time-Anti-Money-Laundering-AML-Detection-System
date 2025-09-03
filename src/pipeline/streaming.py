import json
import threading
import time
from typing import Optional, Callable, Dict, Any
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
from loguru import logger
import pandas as pd

from ..models.data_models import Transaction, AMLAlert

class KafkaTransactionProducer:
    """Kafka producer for financial transactions"""
    
    def __init__(self, 
                 bootstrap_servers: str = "localhost:9092",
                 topic: str = "financial-transactions"):
        """
        Initialize Kafka producer
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Topic to produce messages to
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        self._init_producer()
    
    def _init_producer(self):
        """Initialize Kafka producer with error handling"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retry_backoff_ms=100,
                retries=3
            )
            logger.info(f"Kafka producer initialized for topic: {self.topic}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {str(e)}")
            # For development/testing without Kafka, use a mock producer
            self.producer = MockKafkaProducer()
    
    def send_transaction(self, transaction: Transaction) -> bool:
        """
        Send a transaction to Kafka
        
        Args:
            transaction: Transaction to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Convert transaction to dict
            transaction_dict = transaction.dict()
            
            # Use account ID as key for partitioning
            key = transaction.source_account
            
            # Send message
            future = self.producer.send(
                self.topic,
                key=key,
                value=transaction_dict
            )
            
            # Wait for send to complete (non-blocking in production)
            future.get(timeout=10)
            
            logger.debug(f"Sent transaction {transaction.id} to Kafka")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send transaction {transaction.id}: {str(e)}")
            return False
    
    def close(self):
        """Close the producer"""
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")

class KafkaTransactionConsumer:
    """Kafka consumer for financial transactions with real-time processing"""
    
    def __init__(self, 
                 bootstrap_servers: str = "localhost:9092",
                 topic: str = "financial-transactions",
                 group_id: str = "aml-detector",
                 auto_offset_reset: str = "latest"):
        """
        Initialize Kafka consumer
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Topic to consume from
            group_id: Consumer group ID
            auto_offset_reset: Offset reset policy
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.consumer = None
        self.is_running = False
        self.processing_callback = None
        self._init_consumer()
    
    def _init_consumer(self):
        """Initialize Kafka consumer with error handling"""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                consumer_timeout_ms=1000
            )
            logger.info(f"Kafka consumer initialized for topic: {self.topic}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {str(e)}")
            # For development/testing without Kafka, use a mock consumer
            self.consumer = MockKafkaConsumer()
    
    def set_processing_callback(self, callback: Callable[[Transaction], None]):
        """
        Set callback function to process transactions
        
        Args:
            callback: Function that takes a Transaction and processes it
        """
        self.processing_callback = callback
        logger.info("Processing callback set for consumer")
    
    def start_consuming(self):
        """Start consuming messages in a background thread"""
        if self.is_running:
            logger.warning("Consumer is already running")
            return
        
        self.is_running = True
        consumer_thread = threading.Thread(target=self._consume_loop, daemon=True)
        consumer_thread.start()
        logger.info("Started consuming transactions from Kafka")
    
    def stop_consuming(self):
        """Stop consuming messages"""
        self.is_running = False
        logger.info("Stopped consuming transactions")
    
    def _consume_loop(self):
        """Main consumption loop"""
        while self.is_running:
            try:
                for message in self.consumer:
                    if not self.is_running:
                        break
                    
                    try:
                        # Parse transaction
                        transaction_data = message.value
                        transaction = Transaction(**transaction_data)
                        
                        # Process transaction if callback is set
                        if self.processing_callback:
                            self.processing_callback(transaction)
                        
                        logger.debug(f"Processed transaction {transaction.id}")
                        
                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Error in consumer loop: {str(e)}")
                time.sleep(1)  # Brief pause before retrying
    
    def close(self):
        """Close the consumer"""
        self.stop_consuming()
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")

class KafkaAlertProducer:
    """Kafka producer for AML alerts"""
    
    def __init__(self, 
                 bootstrap_servers: str = "localhost:9092",
                 topic: str = "aml-alerts"):
        """
        Initialize Kafka alert producer
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Topic to produce alerts to
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        self._init_producer()
    
    def _init_producer(self):
        """Initialize Kafka producer with error handling"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retry_backoff_ms=100,
                retries=3
            )
            logger.info(f"Kafka alert producer initialized for topic: {self.topic}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka alert producer: {str(e)}")
            self.producer = MockKafkaProducer()
    
    def send_alert(self, alert: AMLAlert) -> bool:
        """
        Send an AML alert to Kafka
        
        Args:
            alert: AML alert to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Convert alert to dict
            alert_dict = alert.dict()
            
            # Use alert type as key for partitioning
            key = alert.alert_type
            
            # Send message
            future = self.producer.send(
                self.topic,
                key=key,
                value=alert_dict
            )
            
            # Wait for send to complete
            future.get(timeout=10)
            
            logger.info(f"Sent alert {alert.id} to Kafka")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert {alert.id}: {str(e)}")
            return False
    
    def close(self):
        """Close the producer"""
        if self.producer:
            self.producer.close()
            logger.info("Kafka alert producer closed")

# Mock classes for development/testing without Kafka
class MockKafkaProducer:
    """Mock Kafka producer for development without Kafka"""
    
    def send(self, topic, key=None, value=None):
        """Mock send method"""
        logger.debug(f"Mock producer: sent to {topic} with key {key}")
        return MockFuture()
    
    def close(self):
        """Mock close method"""
        logger.debug("Mock producer closed")

class MockKafkaConsumer:
    """Mock Kafka consumer for development without Kafka"""
    
    def __init__(self):
        self.messages = []
    
    def __iter__(self):
        return iter(self.messages)
    
    def close(self):
        """Mock close method"""
        logger.debug("Mock consumer closed")

class MockFuture:
    """Mock future for Kafka send operations"""
    
    def get(self, timeout=None):
        """Mock get method"""
        return True

class StreamingPipeline:
    """Main streaming pipeline orchestrator"""
    
    def __init__(self, 
                 kafka_servers: str = "localhost:9092",
                 transaction_topic: str = "financial-transactions",
                 alert_topic: str = "aml-alerts"):
        """
        Initialize streaming pipeline
        
        Args:
            kafka_servers: Kafka bootstrap servers
            transaction_topic: Topic for incoming transactions
            alert_topic: Topic for outgoing alerts
        """
        self.kafka_servers = kafka_servers
        self.transaction_topic = transaction_topic
        self.alert_topic = alert_topic
        
        # Initialize components
        self.transaction_consumer = KafkaTransactionConsumer(
            bootstrap_servers=kafka_servers,
            topic=transaction_topic
        )
        self.alert_producer = KafkaAlertProducer(
            bootstrap_servers=kafka_servers,
            topic=alert_topic
        )
        
        # Detection components (to be set later)
        self.ml_detector = None
        self.rule_detector = None
        
        logger.info("Streaming pipeline initialized")
    
    def set_detectors(self, ml_detector=None, rule_detector=None):
        """
        Set detection components
        
        Args:
            ml_detector: ML-based anomaly detector
            rule_detector: Rule-based detector
        """
        self.ml_detector = ml_detector
        self.rule_detector = rule_detector
        logger.info("Detectors set for streaming pipeline")
    
    def process_transaction(self, transaction: Transaction):
        """
        Process a single transaction through the AML detection pipeline
        
        Args:
            transaction: Transaction to process
        """
        alerts = []
        
        try:
            # Rule-based detection
            if self.rule_detector:
                rule_alerts = self.rule_detector.analyze_transaction(transaction)
                alerts.extend(rule_alerts)
            
            # ML-based detection (if available)
            if self.ml_detector:
                try:
                    # Convert transaction to DataFrame for ML model
                    df = pd.DataFrame([transaction.dict()])
                    
                    # Get predictions from ML model
                    predictions, scores = self.ml_detector.predict(df)
                    
                    # Create alert if anomaly detected
                    if predictions[0] == -1:  # Anomaly detected (Isolation Forest convention)
                        probability = self.ml_detector.get_anomaly_probability(scores)[0]
                        
                        ml_alert = AMLAlert(
                            id=f"ml_{transaction.id}_{time.time()}",
                            transaction_id=transaction.id,
                            alert_type="ml_anomaly",
                            risk_level="high" if probability > 0.8 else "medium",
                            confidence_score=probability,
                            description=f"ML model detected anomaly with {probability:.2%} confidence",
                            detected_by="isolation_forest",
                            additional_data={"anomaly_score": float(scores[0])}
                        )
                        alerts.append(ml_alert)
                
                except Exception as e:
                    logger.error(f"Error in ML detection: {str(e)}")
            
            # Send alerts
            for alert in alerts:
                self.alert_producer.send_alert(alert)
            
            if alerts:
                logger.info(f"Generated {len(alerts)} alerts for transaction {transaction.id}")
        
        except Exception as e:
            logger.error(f"Error processing transaction {transaction.id}: {str(e)}")
    
    def start(self):
        """Start the streaming pipeline"""
        # Set the processing callback
        self.transaction_consumer.set_processing_callback(self.process_transaction)
        
        # Start consuming
        self.transaction_consumer.start_consuming()
        
        logger.info("Streaming pipeline started")
    
    def stop(self):
        """Stop the streaming pipeline"""
        self.transaction_consumer.stop_consuming()
        self.transaction_consumer.close()
        self.alert_producer.close()
        
        logger.info("Streaming pipeline stopped")