# Smart Industrial Defect Detection - Database Logger
"""
Database logging for defect detection events.
Supports SQLite (edge) and PostgreSQL (cloud).
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DatabaseLogger:
    """
    Database logger for defect detection events.
    Supports SQLite and PostgreSQL.
    """
    
    def __init__(self,
                 db_type: str = 'sqlite',
                 db_path: str = 'data/defect_detection.db',
                 connection_string: str = None):
        """
        Initialize database logger.
        
        Args:
            db_type: Database type ('sqlite' or 'postgresql')
            db_path: Path for SQLite database
            connection_string: PostgreSQL connection string
        """
        self.db_type = db_type
        self.db_path = db_path
        self.connection_string = connection_string
        self.connection = None
        
        self._connect()
        self._create_tables()
        
        logger.info(f"Database logger initialized ({db_type})")
    
    def _connect(self):
        """Connect to database."""
        if self.db_type == 'sqlite':
            import sqlite3
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
        
        elif self.db_type == 'postgresql':
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor
                self.connection = psycopg2.connect(
                    self.connection_string,
                    cursor_factory=RealDictCursor
                )
            except ImportError:
                logger.warning("psycopg2 not installed. Falling back to SQLite.")
                self.db_type = 'sqlite'
                self._connect()
    
    def _create_tables(self):
        """Create database tables."""
        cursor = self.connection.cursor()
        
        # Detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                image_id TEXT,
                decision TEXT,
                confidence REAL,
                anomaly_score REAL,
                latency_ms REAL,
                defects_json TEXT,
                line_id INTEGER,
                batch_id TEXT,
                shift TEXT,
                operator_id TEXT
            )
        ''')
        
        # Defects table (normalized)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS defects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER,
                defect_class TEXT,
                confidence REAL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                severity INTEGER,
                FOREIGN KEY (detection_id) REFERENCES detections (id)
            )
        ''')
        
        # System events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                event_type TEXT,
                event_data TEXT,
                severity TEXT
            )
        ''')
        
        # Hourly statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hourly_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hour_timestamp TEXT,
                total_inspections INTEGER,
                accepts INTEGER,
                rejects INTEGER,
                alerts INTEGER,
                avg_latency_ms REAL,
                defect_rate REAL,
                line_id INTEGER
            )
        ''')
        
        self.connection.commit()
    
    def log_detection(self,
                      image_id: str,
                      decision: str,
                      defects: List[Dict],
                      confidence: float,
                      anomaly_score: float,
                      latency_ms: float,
                      line_id: int = 1,
                      batch_id: str = None,
                      shift: str = None,
                      operator_id: str = None) -> int:
        """
        Log detection result to database.
        
        Args:
            image_id: Image identifier
            decision: Decision (ACCEPT, REJECT, ALERT)
            defects: List of detected defects
            confidence: Overall confidence
            anomaly_score: Anomaly detection score
            latency_ms: Inference latency
            line_id: Production line ID
            batch_id: Batch identifier
            shift: Shift identifier
            operator_id: Operator identifier
        
        Returns:
            Detection ID
        """
        cursor = self.connection.cursor()
        
        # Insert detection
        cursor.execute('''
            INSERT INTO detections 
            (timestamp, image_id, decision, confidence, anomaly_score, 
             latency_ms, defects_json, line_id, batch_id, shift, operator_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            time.time(),
            image_id,
            decision,
            confidence,
            anomaly_score,
            latency_ms,
            json.dumps(defects),
            line_id,
            batch_id,
            shift,
            operator_id
        ))
        
        detection_id = cursor.lastrowid
        
        # Insert individual defects
        for defect in defects:
            bbox = defect.get('bbox', [0, 0, 0, 0])
            cursor.execute('''
                INSERT INTO defects 
                (detection_id, defect_class, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection_id,
                defect.get('classification') or defect.get('detection_class', 'unknown'),
                defect.get('classification_confidence') or defect.get('detection_confidence', 0),
                bbox[0] if len(bbox) > 0 else 0,
                bbox[1] if len(bbox) > 1 else 0,
                bbox[2] if len(bbox) > 2 else 0,
                bbox[3] if len(bbox) > 3 else 0,
                defect.get('severity', 2)
            ))
        
        self.connection.commit()
        return detection_id
    
    def log_event(self,
                  event_type: str,
                  event_data: Dict,
                  severity: str = 'INFO') -> int:
        """
        Log system event.
        
        Args:
            event_type: Type of event
            event_data: Event data dictionary
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
        
        Returns:
            Event ID
        """
        cursor = self.connection.cursor()
        
        cursor.execute('''
            INSERT INTO events (timestamp, event_type, event_data, severity)
            VALUES (?, ?, ?, ?)
        ''', (time.time(), event_type, json.dumps(event_data), severity))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def get_recent_detections(self, limit: int = 100, line_id: int = None) -> List[Dict]:
        """Get recent detections."""
        cursor = self.connection.cursor()
        
        if line_id:
            cursor.execute('''
                SELECT * FROM detections 
                WHERE line_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (line_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM detections 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_statistics(self,
                       hours: int = 24,
                       line_id: int = None) -> Dict:
        """
        Get detection statistics for time period.
        
        Args:
            hours: Number of hours to look back
            line_id: Filter by line ID
        
        Returns:
            Statistics dictionary
        """
        cursor = self.connection.cursor()
        
        since_timestamp = time.time() - (hours * 3600)
        
        if line_id:
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN decision = 'ACCEPT' THEN 1 ELSE 0 END) as accepts,
                    SUM(CASE WHEN decision = 'REJECT' THEN 1 ELSE 0 END) as rejects,
                    SUM(CASE WHEN decision = 'ALERT' THEN 1 ELSE 0 END) as alerts,
                    AVG(latency_ms) as avg_latency,
                    AVG(confidence) as avg_confidence
                FROM detections
                WHERE timestamp > ? AND line_id = ?
            ''', (since_timestamp, line_id))
        else:
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN decision = 'ACCEPT' THEN 1 ELSE 0 END) as accepts,
                    SUM(CASE WHEN decision = 'REJECT' THEN 1 ELSE 0 END) as rejects,
                    SUM(CASE WHEN decision = 'ALERT' THEN 1 ELSE 0 END) as alerts,
                    AVG(latency_ms) as avg_latency,
                    AVG(confidence) as avg_confidence
                FROM detections
                WHERE timestamp > ?
            ''', (since_timestamp,))
        
        row = cursor.fetchone()
        
        total = row['total'] or 0
        accepts = row['accepts'] or 0
        rejects = row['rejects'] or 0
        alerts = row['alerts'] or 0
        
        return {
            'period_hours': hours,
            'total_inspections': total,
            'accepts': accepts,
            'rejects': rejects,
            'alerts': alerts,
            'accept_rate': accepts / max(1, total),
            'reject_rate': rejects / max(1, total),
            'defect_rate': (rejects + alerts) / max(1, total),
            'avg_latency_ms': row['avg_latency'] or 0,
            'avg_confidence': row['avg_confidence'] or 0
        }
    
    def get_defect_distribution(self, hours: int = 24) -> Dict[str, int]:
        """Get defect type distribution."""
        cursor = self.connection.cursor()
        
        since_timestamp = time.time() - (hours * 3600)
        
        cursor.execute('''
            SELECT defect_class, COUNT(*) as count
            FROM defects d
            JOIN detections det ON d.detection_id = det.id
            WHERE det.timestamp > ?
            GROUP BY defect_class
            ORDER BY count DESC
        ''', (since_timestamp,))
        
        rows = cursor.fetchall()
        return {row['defect_class']: row['count'] for row in rows}
    
    def get_hourly_trend(self, hours: int = 24) -> List[Dict]:
        """Get hourly detection trend."""
        cursor = self.connection.cursor()
        
        since_timestamp = time.time() - (hours * 3600)
        
        cursor.execute('''
            SELECT 
                strftime('%Y-%m-%d %H:00', timestamp, 'unixepoch', 'localtime') as hour,
                COUNT(*) as total,
                SUM(CASE WHEN decision = 'REJECT' OR decision = 'ALERT' THEN 1 ELSE 0 END) as defects,
                AVG(latency_ms) as avg_latency
            FROM detections
            WHERE timestamp > ?
            GROUP BY hour
            ORDER BY hour
        ''', (since_timestamp,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
