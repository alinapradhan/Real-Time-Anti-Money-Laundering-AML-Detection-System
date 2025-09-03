import json
import smtplib
import time
from abc import ABC, abstractmethod
from datetime import datetime
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import threading
from loguru import logger

from ..models.data_models import AMLAlert, RiskLevel

class AlertNotifier(ABC):
    """Abstract base class for alert notifiers"""
    
    @abstractmethod
    def send_alert(self, alert: AMLAlert) -> bool:
        """Send an alert notification"""
        pass

class EmailNotifier(AlertNotifier):
    """Email-based alert notifier"""
    
    def __init__(self, 
                 smtp_server: str = "localhost",
                 smtp_port: int = 587,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 from_email: str = "aml-system@company.com",
                 to_emails: List[str] = None):
        """
        Initialize email notifier
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: From email address
            to_emails: List of recipient email addresses
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails or ["compliance@company.com"]
    
    def send_alert(self, alert: AMLAlert) -> bool:
        """
        Send alert via email
        
        Args:
            alert: AML alert to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)
            msg['Subject'] = f"AML Alert: {alert.risk_level.upper()} - {alert.alert_type}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            # Send email (mock implementation for demo)
            logger.info(f"Mock email sent for alert {alert.id} to {self.to_emails}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert {alert.id}: {str(e)}")
            return False
    
    def _create_email_body(self, alert: AMLAlert) -> str:
        """Create HTML email body for the alert"""
        risk_color = {
            RiskLevel.LOW: "#28a745",
            RiskLevel.MEDIUM: "#ffc107", 
            RiskLevel.HIGH: "#fd7e14",
            RiskLevel.CRITICAL: "#dc3545"
        }.get(alert.risk_level, "#6c757d")
        
        return f"""
        <html>
        <body>
            <h2 style="color: {risk_color};">AML Alert - {alert.risk_level.upper()}</h2>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr><td><strong>Alert ID:</strong></td><td>{alert.id}</td></tr>
                <tr><td><strong>Transaction ID:</strong></td><td>{alert.transaction_id}</td></tr>
                <tr><td><strong>Alert Type:</strong></td><td>{alert.alert_type}</td></tr>
                <tr><td><strong>Risk Level:</strong></td><td style="color: {risk_color};">{alert.risk_level.upper()}</td></tr>
                <tr><td><strong>Confidence:</strong></td><td>{alert.confidence_score:.2%}</td></tr>
                <tr><td><strong>Description:</strong></td><td>{alert.description}</td></tr>
                <tr><td><strong>Detected By:</strong></td><td>{alert.detected_by}</td></tr>
                <tr><td><strong>Timestamp:</strong></td><td>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
            </table>
            
            <h3>Additional Information</h3>
            <pre>{json.dumps(alert.additional_data, indent=2)}</pre>
            
            <p><strong>Please review this alert and take appropriate action.</strong></p>
        </body>
        </html>
        """

class LogNotifier(AlertNotifier):
    """Log-based alert notifier"""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize log notifier
        
        Args:
            log_file: Optional log file path
        """
        self.log_file = log_file
    
    def send_alert(self, alert: AMLAlert) -> bool:
        """
        Log the alert
        
        Args:
            alert: AML alert to log
            
        Returns:
            Always returns True
        """
        alert_data = {
            "alert_id": alert.id,
            "transaction_id": alert.transaction_id,
            "alert_type": alert.alert_type,
            "risk_level": alert.risk_level,
            "confidence": alert.confidence_score,
            "description": alert.description,
            "detected_by": alert.detected_by,
            "timestamp": alert.timestamp.isoformat(),
            "additional_data": alert.additional_data
        }
        
        if alert.risk_level == RiskLevel.CRITICAL:
            logger.critical(f"CRITICAL AML ALERT: {json.dumps(alert_data)}")
        elif alert.risk_level == RiskLevel.HIGH:
            logger.error(f"HIGH RISK AML ALERT: {json.dumps(alert_data)}")
        elif alert.risk_level == RiskLevel.MEDIUM:
            logger.warning(f"MEDIUM RISK AML ALERT: {json.dumps(alert_data)}")
        else:
            logger.info(f"LOW RISK AML ALERT: {json.dumps(alert_data)}")
        
        return True

class DashboardNotifier(AlertNotifier):
    """Dashboard/real-time notification for alerts"""
    
    def __init__(self):
        """Initialize dashboard notifier"""
        self.recent_alerts = deque(maxlen=1000)
        self.alert_stats = defaultdict(int)
        self.lock = threading.Lock()
    
    def send_alert(self, alert: AMLAlert) -> bool:
        """
        Store alert for dashboard display
        
        Args:
            alert: AML alert to store
            
        Returns:
            Always returns True
        """
        with self.lock:
            # Add to recent alerts
            self.recent_alerts.append(alert)
            
            # Update statistics
            self.alert_stats[alert.risk_level.value] += 1
            self.alert_stats['total'] += 1
            self.alert_stats[f"type_{alert.alert_type}"] += 1
        
        logger.debug(f"Alert {alert.id} stored for dashboard")
        return True
    
    def get_recent_alerts(self, limit: int = 100) -> List[AMLAlert]:
        """Get recent alerts for dashboard display"""
        with self.lock:
            return list(self.recent_alerts)[-limit:]
    
    def get_alert_stats(self) -> Dict[str, int]:
        """Get alert statistics"""
        with self.lock:
            return dict(self.alert_stats)

class AlertManager:
    """Main alert management system"""
    
    def __init__(self):
        """Initialize alert manager"""
        self.notifiers = []
        self.alert_history = deque(maxlen=10000)
        self.duplicate_prevention = {}
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.Lock()
        
        # Initialize default notifiers
        self.log_notifier = LogNotifier()
        self.dashboard_notifier = DashboardNotifier()
        self.email_notifier = EmailNotifier()
        
        self.add_notifier(self.log_notifier)
        self.add_notifier(self.dashboard_notifier)
        
        logger.info("Alert manager initialized")
    
    def add_notifier(self, notifier: AlertNotifier):
        """Add a notifier to the alert manager"""
        self.notifiers.append(notifier)
        logger.info(f"Added notifier: {type(notifier).__name__}")
    
    def process_alert(self, alert: AMLAlert) -> bool:
        """
        Process an alert through all notifiers
        
        Args:
            alert: AML alert to process
            
        Returns:
            True if processed successfully, False otherwise
        """
        try:
            # Check for duplicates
            if self._is_duplicate(alert):
                logger.debug(f"Duplicate alert {alert.id} ignored")
                return False
            
            # Check rate limits
            if self._is_rate_limited(alert):
                logger.warning(f"Rate limited alert {alert.id}")
                return False
            
            # Store in history
            with self.lock:
                self.alert_history.append(alert)
            
            # Send through all notifiers
            success_count = 0
            for notifier in self.notifiers:
                try:
                    if notifier.send_alert(alert):
                        success_count += 1
                except Exception as e:
                    logger.error(f"Notifier {type(notifier).__name__} failed: {str(e)}")
            
            # Add email notification for high/critical alerts
            if alert.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                try:
                    self.email_notifier.send_alert(alert)
                except Exception as e:
                    logger.error(f"Email notification failed: {str(e)}")
            
            logger.info(f"Alert {alert.id} processed by {success_count}/{len(self.notifiers)} notifiers")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error processing alert {alert.id}: {str(e)}")
            return False
    
    def _is_duplicate(self, alert: AMLAlert) -> bool:
        """Check if alert is a duplicate"""
        # Simple duplicate detection based on transaction ID and alert type
        key = f"{alert.transaction_id}_{alert.alert_type}"
        current_time = time.time()
        
        if key in self.duplicate_prevention:
            last_time = self.duplicate_prevention[key]
            if current_time - last_time < 300:  # 5 minutes
                return True
        
        self.duplicate_prevention[key] = current_time
        return False
    
    def _is_rate_limited(self, alert: AMLAlert) -> bool:
        """Check if alert should be rate limited"""
        current_time = time.time()
        
        # Rate limit per alert type
        rate_key = alert.alert_type
        self.rate_limits[rate_key].append(current_time)
        
        # Remove old entries (older than 1 hour)
        while (self.rate_limits[rate_key] and 
               current_time - self.rate_limits[rate_key][0] > 3600):
            self.rate_limits[rate_key].popleft()
        
        # Check if rate limit exceeded (max 50 alerts per hour per type)
        if len(self.rate_limits[rate_key]) > 50:
            return True
        
        return False
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display"""
        return {
            'recent_alerts': self.dashboard_notifier.get_recent_alerts(),
            'stats': self.dashboard_notifier.get_alert_stats(),
            'total_alerts_processed': len(self.alert_history)
        }
    
    def get_alert_history(self, limit: int = 100) -> List[AMLAlert]:
        """Get recent alert history"""
        with self.lock:
            return list(self.alert_history)[-limit:]