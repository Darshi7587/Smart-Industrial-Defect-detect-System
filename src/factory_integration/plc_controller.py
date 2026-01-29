# Smart Industrial Defect Detection - PLC Controller
"""
PLC (Programmable Logic Controller) integration for factory automation.
Uses Modbus TCP/IP protocol for industrial communication.
"""

import time
import struct
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PLCSignal:
    """PLC signal definition."""
    address: int
    value: int
    timestamp: float
    signal_type: str
    duration_ms: int = 50


class PLCController:
    """
    PLC controller for factory automation integration.
    Communicates via Modbus TCP/IP protocol.
    """
    
    def __init__(self,
                 ip: str = "192.168.1.50",
                 port: int = 502,
                 timeout: float = 5.0,
                 simulate: bool = True):
        """
        Initialize PLC controller.
        
        Args:
            ip: PLC IP address
            port: Modbus port (usually 502)
            timeout: Connection timeout
            simulate: Run in simulation mode (no real PLC)
        """
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.simulate = simulate
        self.connected = False
        self.client = None
        
        # Signal addresses (configurable per PLC setup)
        self.addresses = {
            'ACCEPT': 100,
            'REJECT': 101,
            'ALERT': 102,
            'MANUAL_REVIEW': 103,
            'CONVEYOR_STOP': 104,
            'CONVEYOR_START': 105,
            'REJECT_SOLENOID': 106,
            'ALARM': 107,
            'HEARTBEAT': 108
        }
        
        # Signal log
        self.signal_log = []
        
        if not simulate:
            self._connect()
        else:
            self.connected = True
            logger.info("PLC Controller initialized in simulation mode")
    
    def _connect(self):
        """Connect to PLC via Modbus TCP."""
        try:
            from pymodbus.client import ModbusTcpClient
            
            self.client = ModbusTcpClient(
                host=self.ip,
                port=self.port,
                timeout=self.timeout
            )
            
            if self.client.connect():
                self.connected = True
                logger.info(f"Connected to PLC at {self.ip}:{self.port}")
            else:
                logger.error(f"Failed to connect to PLC at {self.ip}:{self.port}")
                self.connected = False
        
        except ImportError:
            logger.warning("pymodbus not installed. Running in simulation mode.")
            self.simulate = True
            self.connected = True
        except Exception as e:
            logger.error(f"PLC connection error: {e}")
            self.connected = False
    
    def disconnect(self):
        """Disconnect from PLC."""
        if self.client and not self.simulate:
            self.client.close()
            self.connected = False
            logger.info("Disconnected from PLC")
    
    def send_signal(self,
                    signal_type: str,
                    value: int = 1,
                    duration_ms: int = 50) -> bool:
        """
        Send signal to PLC.
        
        Args:
            signal_type: Signal type (ACCEPT, REJECT, ALERT, etc.)
            value: Signal value
            duration_ms: Pulse duration in milliseconds
        
        Returns:
            True if signal sent successfully
        """
        if signal_type not in self.addresses:
            logger.error(f"Unknown signal type: {signal_type}")
            return False
        
        address = self.addresses[signal_type]
        timestamp = time.time()
        
        # Create signal record
        signal = PLCSignal(
            address=address,
            value=value,
            timestamp=timestamp,
            signal_type=signal_type,
            duration_ms=duration_ms
        )
        
        # Log signal
        self.signal_log.append(signal)
        
        if self.simulate:
            logger.debug(f"[SIM] PLC Signal: {signal_type} = {value} at address {address}")
            return True
        
        if not self.connected:
            logger.error("Not connected to PLC")
            return False
        
        try:
            # Write to coil
            result = self.client.write_coil(address, value == 1)
            
            if result.isError():
                logger.error(f"Failed to write PLC signal: {result}")
                return False
            
            # Handle pulse duration (turn off after duration)
            if duration_ms > 0 and value == 1:
                time.sleep(duration_ms / 1000)
                self.client.write_coil(address, False)
            
            logger.debug(f"PLC Signal sent: {signal_type} = {value} at address {address}")
            return True
        
        except Exception as e:
            logger.error(f"PLC write error: {e}")
            return False
    
    def read_signal(self, signal_type: str) -> Optional[int]:
        """
        Read signal from PLC.
        
        Args:
            signal_type: Signal type to read
        
        Returns:
            Signal value or None if error
        """
        if signal_type not in self.addresses:
            logger.error(f"Unknown signal type: {signal_type}")
            return None
        
        address = self.addresses[signal_type]
        
        if self.simulate:
            return 0  # Simulated value
        
        if not self.connected:
            logger.error("Not connected to PLC")
            return None
        
        try:
            result = self.client.read_coils(address, 1)
            if result.isError():
                logger.error(f"Failed to read PLC signal: {result}")
                return None
            return 1 if result.bits[0] else 0
        
        except Exception as e:
            logger.error(f"PLC read error: {e}")
            return None
    
    def activate_reject_solenoid(self, duration_ms: int = 50) -> bool:
        """
        Activate reject solenoid to divert defective product.
        
        Args:
            duration_ms: Solenoid activation duration
        
        Returns:
            True if successful
        """
        logger.info(f"Activating reject solenoid for {duration_ms}ms")
        return self.send_signal('REJECT_SOLENOID', 1, duration_ms)
    
    def stop_conveyor(self) -> bool:
        """Stop conveyor belt."""
        logger.info("Stopping conveyor")
        return self.send_signal('CONVEYOR_STOP', 1)
    
    def start_conveyor(self) -> bool:
        """Start conveyor belt."""
        logger.info("Starting conveyor")
        return self.send_signal('CONVEYOR_START', 1)
    
    def trigger_alarm(self, alarm_on: bool = True) -> bool:
        """
        Trigger or disable alarm.
        
        Args:
            alarm_on: True to turn on, False to turn off
        
        Returns:
            True if successful
        """
        logger.info(f"Alarm {'ON' if alarm_on else 'OFF'}")
        return self.send_signal('ALARM', 1 if alarm_on else 0, 0)
    
    def send_heartbeat(self) -> bool:
        """Send heartbeat signal to indicate system is alive."""
        return self.send_signal('HEARTBEAT', 1, 100)
    
    def handle_decision(self, decision: str, severity: int = 2) -> Dict:
        """
        Handle decision from AI system.
        
        Args:
            decision: Decision string (ACCEPT, REJECT, ALERT, MANUAL_REVIEW)
            severity: Defect severity (1-4)
        
        Returns:
            Dictionary with action results
        """
        actions_taken = []
        
        if decision == 'ACCEPT':
            self.send_signal('ACCEPT', 1)
            actions_taken.append('ACCEPT signal sent')
        
        elif decision == 'REJECT':
            self.send_signal('REJECT', 1)
            self.activate_reject_solenoid(duration_ms=50)
            actions_taken.append('REJECT signal sent')
            actions_taken.append('Reject solenoid activated')
            
            if severity >= 4:  # Critical defect
                self.trigger_alarm(True)
                actions_taken.append('Alarm triggered')
        
        elif decision == 'ALERT':
            self.send_signal('ALERT', 1)
            actions_taken.append('ALERT signal sent')
            
            if severity >= 3:
                self.trigger_alarm(True)
                actions_taken.append('Alarm triggered')
        
        elif decision == 'MANUAL_REVIEW':
            self.send_signal('MANUAL_REVIEW', 1)
            self.stop_conveyor()
            actions_taken.append('MANUAL_REVIEW signal sent')
            actions_taken.append('Conveyor stopped')
        
        return {
            'decision': decision,
            'severity': severity,
            'actions': actions_taken,
            'timestamp': time.time()
        }
    
    def get_signal_log(self, limit: int = 100) -> list:
        """Get recent signal log."""
        return self.signal_log[-limit:]
    
    def get_status(self) -> Dict:
        """Get PLC controller status."""
        return {
            'connected': self.connected,
            'simulate': self.simulate,
            'ip': self.ip,
            'port': self.port,
            'signals_sent': len(self.signal_log),
            'addresses': self.addresses
        }
