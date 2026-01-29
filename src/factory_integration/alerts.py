# Smart Industrial Defect Detection - Alert Manager
"""
Alert and notification system for defect detection events.
"""

import time
import json
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import threading
from queue import Queue

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


@dataclass
class Alert:
    """Alert definition."""
    alert_id: str
    timestamp: float
    severity: AlertSeverity
    title: str
    message: str
    source: str
    data: Dict
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None


class AlertManager:
    """
    Alert manager for defect detection system.
    Handles alerts, notifications, and escalations.
    """
    
    def __init__(self,
                 email_enabled: bool = False,
                 slack_enabled: bool = False,
                 webhook_url: str = None):
        """
        Initialize alert manager.
        
        Args:
            email_enabled: Enable email notifications
            slack_enabled: Enable Slack notifications
            webhook_url: Webhook URL for notifications
        """
        self.email_enabled = email_enabled
        self.slack_enabled = slack_enabled
        self.webhook_url = webhook_url
        
        # Alert storage
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        
        # Alert thresholds
        self.thresholds = {
            'defect_rate': 0.1,  # 10% defect rate triggers alert
            'latency': 150,  # 150ms latency triggers warning
            'consecutive_rejects': 5,  # 5 consecutive rejects triggers alert
            'anomaly_spike': 3.0  # Anomaly score > 3 triggers alert
        }
        
        # Counters for threshold checking
        self.consecutive_rejects = 0
        self.recent_defect_rate = 0.0
        
        # Callback handlers
        self.handlers: List[Callable] = []
        
        # Alert ID counter
        self._alert_counter = 0
        
        logger.info("Alert manager initialized")
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"ALERT_{int(time.time())}_{self._alert_counter}"
    
    def create_alert(self,
                     severity: AlertSeverity,
                     title: str,
                     message: str,
                     source: str = 'system',
                     data: Dict = None) -> Alert:
        """
        Create and register new alert.
        
        Args:
            severity: Alert severity
            title: Alert title
            message: Alert message
            source: Alert source
            data: Additional data
        
        Returns:
            Created alert
        """
        alert = Alert(
            alert_id=self._generate_alert_id(),
            timestamp=time.time(),
            severity=severity,
            title=title,
            message=message,
            source=source,
            data=data or {}
        )
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Trigger handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        # Send notifications for high severity
        if severity.value >= AlertSeverity.ERROR.value:
            self._send_notifications(alert)
        
        logger.warning(f"Alert created: [{severity.name}] {title} - {message}")
        
        return alert
    
    def acknowledge_alert(self, 
                          alert_id: str, 
                          acknowledged_by: str = 'operator') -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: Who acknowledged
        
        Returns:
            True if acknowledged successfully
        """
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = time.time()
                
                # Remove from active alerts
                self.active_alerts.remove(alert)
                
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        
        return False
    
    def check_defect_rate(self, 
                          total: int, 
                          defects: int, 
                          window: str = 'hourly'):
        """
        Check if defect rate exceeds threshold.
        
        Args:
            total: Total inspections
            defects: Number of defects
            window: Time window description
        """
        if total == 0:
            return
        
        rate = defects / total
        self.recent_defect_rate = rate
        
        if rate > self.thresholds['defect_rate']:
            self.create_alert(
                severity=AlertSeverity.WARNING,
                title="High Defect Rate",
                message=f"Defect rate {rate:.1%} exceeds threshold {self.thresholds['defect_rate']:.1%}",
                source='defect_monitor',
                data={'rate': rate, 'total': total, 'defects': defects, 'window': window}
            )
    
    def check_latency(self, latency_ms: float):
        """
        Check if latency exceeds threshold.
        
        Args:
            latency_ms: Current latency in milliseconds
        """
        if latency_ms > self.thresholds['latency']:
            self.create_alert(
                severity=AlertSeverity.WARNING,
                title="High Latency",
                message=f"Inference latency {latency_ms:.1f}ms exceeds threshold {self.thresholds['latency']}ms",
                source='latency_monitor',
                data={'latency_ms': latency_ms}
            )
    
    def check_consecutive_rejects(self, decision: str):
        """
        Check for consecutive rejects.
        
        Args:
            decision: Latest decision
        """
        if decision == 'REJECT':
            self.consecutive_rejects += 1
            
            if self.consecutive_rejects >= self.thresholds['consecutive_rejects']:
                self.create_alert(
                    severity=AlertSeverity.ERROR,
                    title="Consecutive Rejects",
                    message=f"{self.consecutive_rejects} consecutive products rejected",
                    source='quality_monitor',
                    data={'count': self.consecutive_rejects}
                )
        else:
            self.consecutive_rejects = 0
    
    def check_anomaly(self, anomaly_score: float):
        """
        Check if anomaly score exceeds threshold.
        
        Args:
            anomaly_score: Anomaly detection score
        """
        if anomaly_score > self.thresholds['anomaly_spike']:
            self.create_alert(
                severity=AlertSeverity.WARNING,
                title="Anomaly Detected",
                message=f"Anomaly score {anomaly_score:.2f} exceeds threshold",
                source='anomaly_detector',
                data={'score': anomaly_score}
            )
    
    def add_handler(self, handler: Callable):
        """Add alert handler callback."""
        self.handlers.append(handler)
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for alert."""
        if self.webhook_url:
            self._send_webhook(alert)
        
        if self.email_enabled:
            self._send_email(alert)
        
        if self.slack_enabled:
            self._send_slack(alert)
    
    def _send_webhook(self, alert: Alert):
        """Send webhook notification."""
        try:
            import requests
            
            payload = {
                'alert_id': alert.alert_id,
                'severity': alert.severity.name,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp,
                'data': alert.data
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5
            )
            
            if response.status_code != 200:
                logger.error(f"Webhook failed: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Webhook error: {e}")
    
    def _send_email(self, alert: Alert):
        """Send email notification (placeholder)."""
        logger.info(f"Email notification: {alert.title}")
        # Implement email sending with smtplib
    
    def _send_slack(self, alert: Alert):
        """Send Slack notification (placeholder)."""
        logger.info(f"Slack notification: {alert.title}")
        # Implement Slack webhook
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unacknowledged) alerts."""
        return self.active_alerts.copy()
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity level."""
        return [a for a in self.active_alerts if a.severity == severity]
    
    def update_thresholds(self, **kwargs):
        """Update alert thresholds."""
        for key, value in kwargs.items():
            if key in self.thresholds:
                self.thresholds[key] = value
                logger.info(f"Threshold updated: {key} = {value}")
    
    def clear_alerts(self):
        """Clear all active alerts."""
        self.active_alerts = []
        logger.info("All active alerts cleared")
