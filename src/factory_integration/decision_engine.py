# Smart Industrial Defect Detection - Decision Engine
"""
Decision engine for accept/reject/alert logic.
Implements configurable business rules for defect handling.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Decision(Enum):
    """Decision types."""
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    ALERT = "ALERT"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class DefectSeverity(Enum):
    """Defect severity levels."""
    MINOR = 1
    MODERATE = 2
    MAJOR = 3
    CRITICAL = 4


@dataclass
class DecisionResult:
    """Result of decision engine."""
    decision: Decision
    confidence: float
    reasons: List[str]
    severity: DefectSeverity
    action_code: int
    timestamp: float
    metadata: Dict


class DecisionEngine:
    """
    Decision engine for manufacturing defect detection.
    Implements configurable rules for accept/reject decisions.
    """
    
    def __init__(self,
                 detection_threshold: float = 0.5,
                 classification_threshold: float = 0.85,
                 anomaly_threshold: float = 2.0,
                 reject_severity_threshold: int = 2,
                 manual_review_threshold: float = 0.5):
        """
        Initialize decision engine.
        
        Args:
            detection_threshold: Minimum confidence for detection
            classification_threshold: Minimum confidence for classification
            anomaly_threshold: Anomaly score threshold (std devs)
            reject_severity_threshold: Minimum severity for rejection
            manual_review_threshold: Confidence below which to flag for review
        """
        self.detection_threshold = detection_threshold
        self.classification_threshold = classification_threshold
        self.anomaly_threshold = anomaly_threshold
        self.reject_severity_threshold = reject_severity_threshold
        self.manual_review_threshold = manual_review_threshold
        
        # Defect severity mapping
        self.defect_severity_map = {
            'scratch': DefectSeverity.MINOR,
            'contamination': DefectSeverity.MINOR,
            'dent': DefectSeverity.MODERATE,
            'crack': DefectSeverity.MAJOR,
            'missing_component': DefectSeverity.CRITICAL
        }
        
        # Action codes for PLC
        self.action_codes = {
            Decision.ACCEPT: 0x01,
            Decision.REJECT: 0x02,
            Decision.ALERT: 0x03,
            Decision.MANUAL_REVIEW: 0x04
        }
        
        # Statistics
        self.stats = {
            'total_decisions': 0,
            'accepts': 0,
            'rejects': 0,
            'alerts': 0,
            'manual_reviews': 0
        }
        
        logger.info("Decision engine initialized")
    
    def make_decision(self,
                      defects: List[Dict],
                      anomaly_score: float = 0.0,
                      max_confidence: float = 0.0) -> DecisionResult:
        """
        Make accept/reject/alert decision based on defects.
        
        Args:
            defects: List of detected defects
            anomaly_score: Anomaly detection score
            max_confidence: Maximum detection confidence
        
        Returns:
            DecisionResult with decision and metadata
        """
        timestamp = time.time()
        reasons = []
        
        # No defects detected
        if not defects:
            if anomaly_score > self.anomaly_threshold:
                decision = Decision.ALERT
                reasons.append(f"Anomaly detected: score={anomaly_score:.2f}")
                severity = DefectSeverity.MODERATE
            else:
                decision = Decision.ACCEPT
                reasons.append("No defects detected")
                severity = DefectSeverity.MINOR
            
            return self._create_result(
                decision, 1.0 - min(1.0, anomaly_score / 10),
                reasons, severity, timestamp, {'anomaly_score': anomaly_score}
            )
        
        # Analyze defects
        max_severity = DefectSeverity.MINOR
        high_confidence_defects = []
        low_confidence_defects = []
        
        for defect in defects:
            # Get severity
            defect_class = defect.get('classification') or defect.get('detection_class', 'unknown')
            severity = self.defect_severity_map.get(defect_class, DefectSeverity.MODERATE)
            
            if severity.value > max_severity.value:
                max_severity = severity
            
            # Categorize by confidence
            confidence = defect.get('classification_confidence') or defect.get('detection_confidence', 0)
            if confidence >= self.classification_threshold:
                high_confidence_defects.append(defect)
            elif confidence >= self.manual_review_threshold:
                low_confidence_defects.append(defect)
        
        # Decision logic
        if high_confidence_defects:
            if max_severity.value >= self.reject_severity_threshold:
                decision = Decision.REJECT
                reasons.append(f"High confidence {max_severity.name} defect detected")
                for d in high_confidence_defects:
                    reasons.append(f"- {d.get('classification', d.get('detection_class'))}: "
                                  f"{d.get('classification_confidence', d.get('detection_confidence')):.2f}")
            else:
                # Minor defects - might still accept
                decision = Decision.ALERT
                reasons.append(f"Minor defects detected, flagged for review")
        
        elif low_confidence_defects:
            decision = Decision.MANUAL_REVIEW
            reasons.append("Low confidence detections require manual review")
        
        else:
            decision = Decision.ACCEPT
            reasons.append("All detections below threshold")
        
        # Check anomaly score as additional signal
        if anomaly_score > self.anomaly_threshold:
            if decision == Decision.ACCEPT:
                decision = Decision.ALERT
            reasons.append(f"Elevated anomaly score: {anomaly_score:.2f}")
        
        # Calculate overall confidence
        if defects:
            confidences = [d.get('classification_confidence', d.get('detection_confidence', 0)) 
                          for d in defects]
            confidence = sum(confidences) / len(confidences)
        else:
            confidence = 1.0
        
        return self._create_result(
            decision, confidence, reasons, max_severity, timestamp,
            {
                'num_defects': len(defects),
                'high_confidence_count': len(high_confidence_defects),
                'low_confidence_count': len(low_confidence_defects),
                'anomaly_score': anomaly_score
            }
        )
    
    def _create_result(self,
                       decision: Decision,
                       confidence: float,
                       reasons: List[str],
                       severity: DefectSeverity,
                       timestamp: float,
                       metadata: Dict) -> DecisionResult:
        """Create decision result and update stats."""
        # Update statistics
        self.stats['total_decisions'] += 1
        if decision == Decision.ACCEPT:
            self.stats['accepts'] += 1
        elif decision == Decision.REJECT:
            self.stats['rejects'] += 1
        elif decision == Decision.ALERT:
            self.stats['alerts'] += 1
        else:
            self.stats['manual_reviews'] += 1
        
        return DecisionResult(
            decision=decision,
            confidence=confidence,
            reasons=reasons,
            severity=severity,
            action_code=self.action_codes[decision],
            timestamp=timestamp,
            metadata=metadata
        )
    
    def update_thresholds(self,
                          detection_threshold: float = None,
                          classification_threshold: float = None,
                          anomaly_threshold: float = None,
                          reject_severity_threshold: int = None):
        """Update decision thresholds."""
        if detection_threshold is not None:
            self.detection_threshold = detection_threshold
        if classification_threshold is not None:
            self.classification_threshold = classification_threshold
        if anomaly_threshold is not None:
            self.anomaly_threshold = anomaly_threshold
        if reject_severity_threshold is not None:
            self.reject_severity_threshold = reject_severity_threshold
        
        logger.info(f"Thresholds updated: detection={self.detection_threshold}, "
                   f"classification={self.classification_threshold}, "
                   f"anomaly={self.anomaly_threshold}")
    
    def get_statistics(self) -> Dict:
        """Get decision statistics."""
        total = self.stats['total_decisions']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'accept_rate': self.stats['accepts'] / total,
            'reject_rate': self.stats['rejects'] / total,
            'alert_rate': self.stats['alerts'] / total,
            'manual_review_rate': self.stats['manual_reviews'] / total
        }
    
    def reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            'total_decisions': 0,
            'accepts': 0,
            'rejects': 0,
            'alerts': 0,
            'manual_reviews': 0
        }
