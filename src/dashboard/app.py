# Smart Industrial Defect Detection - Streamlit Dashboard
"""
Real-time monitoring dashboard for defect detection system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
import sys
import sqlite3

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Database path
DB_PATH = Path(__file__).parent.parent.parent / "data" / "defect_detection.db"


def load_data_from_db():
    """Load real detection data from database."""
    if not DB_PATH.exists():
        return None
    
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("""
            SELECT * FROM detections 
            ORDER BY timestamp DESC 
            LIMIT 1000
        """, conn)
        conn.close()
        
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    except Exception as e:
        st.error(f"Database error: {e}")
    
    return None


def create_dashboard():
    """Create Streamlit dashboard."""
    
    st.set_page_config(
        page_title="Defect Detection Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .status-accept { color: #28a745; }
        .status-reject { color: #dc3545; }
        .status-alert { color: #ffc107; }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔍 Smart Industrial Defect Detection")
        st.caption("Real-time AI-powered quality control monitoring")
    with col2:
        st.metric("System Status", "🟢 Online")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Line selection
    line_id = st.sidebar.selectbox(
        "Production Line",
        options=[1, 2, 3, 4],
        index=0
    )
    
    # Time range
    time_range = st.sidebar.selectbox(
        "Time Range",
        options=["Last Hour", "Last 8 Hours", "Last 24 Hours", "Last Week"],
        index=2
    )
    
    # Auto refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=False)
    
    if auto_refresh:
        st.sidebar.info("Dashboard refreshing every 5 seconds")
    
    # Threshold settings
    st.sidebar.subheader("Detection Thresholds")
    detection_threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5)
    anomaly_threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 5.0, 2.0)
    
    # Main content
    # Load real data from database
    df = load_data_from_db()
    use_real_data = df is not None and len(df) > 0
    
    if use_real_data:
        st.success("📊 Using REAL detection data from database")
    else:
        st.warning("⚠️ No real data found. Run `python scripts/run_demo.py` to generate data.")
        np.random.seed(42)
    
    # KPI Metrics
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if use_real_data:
        total_inspected = len(df)
        defect_count = len(df[df['defect_type'] != 'none'])
        accept_count = len(df[df['decision'] == 'ACCEPT'])
        reject_count = len(df[df['decision'] == 'REJECT'])
        defect_rate = defect_count / total_inspected if total_inspected > 0 else 0
        accept_rate = accept_count / total_inspected if total_inspected > 0 else 0
        avg_latency = df['latency_ms'].mean()
    else:
        total_inspected = np.random.randint(5000, 10000)
        defect_rate = np.random.uniform(0.02, 0.08)
        accept_rate = 1 - defect_rate - np.random.uniform(0.01, 0.02)
        avg_latency = np.random.uniform(25, 45)
    
    with col1:
        st.metric(
            "Total Inspected",
            f"{total_inspected:,}",
            delta=f"+{total_inspected}" if use_real_data else f"+{np.random.randint(100, 500)}"
        )
    
    with col2:
        st.metric(
            "Defect Rate",
            f"{defect_rate:.1%}",
            delta=f"{defect_rate:.2%}",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Accept Rate",
            f"{accept_rate:.1%}",
            delta=f"{accept_rate:.1%}"
        )
    
    with col4:
        st.metric(
            "Avg Latency",
            f"{avg_latency:.1f}ms",
            delta="-2.1ms"
        )
    
    with col5:
        throughput = int(3600 / (avg_latency / 1000)) if avg_latency > 0 else 1000
        st.metric(
            "Throughput/hr",
            f"{throughput}",
            delta="+50"
        )
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Defect Trend (24 Hours)")
        
        if use_real_data:
            # Group by hour
            df_hourly = df.copy()
            df_hourly['hour'] = df_hourly['timestamp'].dt.floor('H')
            hourly_stats = df_hourly.groupby('hour').agg({
                'image_id': 'count',
                'defect_type': lambda x: (x != 'none').sum()
            }).rename(columns={'image_id': 'Total', 'defect_type': 'Defects'})
            hourly_stats['Rate'] = hourly_stats['Defects'] / hourly_stats['Total'] * 100
            hourly_stats = hourly_stats.reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['Defects'],
                mode='lines+markers',
                name='Defects',
                line=dict(color='#dc3545')
            ))
            fig.add_trace(go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['Rate'],
                mode='lines',
                name='Defect Rate %',
                yaxis='y2',
                line=dict(color='#007bff', dash='dash')
            ))
        else:
            hours = pd.date_range(end=datetime.now(), periods=24, freq='H')
            defects = np.random.randint(5, 50, size=24)
            inspections = np.random.randint(200, 400, size=24)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours,
                y=defects,
                mode='lines+markers',
                name='Defects',
                line=dict(color='#dc3545')
            ))
            fig.add_trace(go.Scatter(
                x=hours,
                y=defects / inspections * 100,
                mode='lines',
                name='Defect Rate %',
                yaxis='y2',
                line=dict(color='#007bff', dash='dash')
            ))
        
        fig.update_layout(
            yaxis=dict(title='Defects'),
            yaxis2=dict(title='Rate %', overlaying='y', side='right'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🥧 Defect Distribution")
        
        if use_real_data:
            defect_counts = df[df['defect_type'] != 'none']['defect_type'].value_counts()
            if len(defect_counts) > 0:
                fig = px.pie(
                    values=defect_counts.values,
                    names=defect_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hole=0.4
                )
            else:
                fig = px.pie(values=[1], names=['No Defects'], hole=0.4)
        else:
            defect_types = ['Scratch', 'Crack', 'Dent', 'Missing Part', 'Contamination']
            defect_counts = [45, 28, 35, 12, 20]
            
            fig = px.pie(
                values=defect_counts,
                names=defect_types,
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4
            )
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏱️ Latency Distribution")
        
        if use_real_data:
            latencies = df['latency_ms'].values
        else:
            latencies = np.random.exponential(scale=30, size=1000) + 10
            latencies = np.clip(latencies, 10, 150)
        
        fig = px.histogram(
            x=latencies,
            nbins=50,
            labels={'x': 'Latency (ms)', 'y': 'Count'},
            color_discrete_sequence=['#007bff']
        )
        fig.add_vline(x=50, line_dash="dash", line_color="red", 
                     annotation_text="Target: 50ms")
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Decision Distribution")
        
        if use_real_data:
            decision_counts = df['decision'].value_counts()
            fig = px.bar(
                x=decision_counts.index,
                y=decision_counts.values,
                labels={'x': 'Decision', 'y': 'Count'},
                color=decision_counts.index,
                color_discrete_map={'ACCEPT': '#28a745', 'REJECT': '#dc3545', 'ALERT': '#ffc107'}
            )
        else:
            hours = list(range(24))
            throughputs = np.random.randint(600, 1200, size=24)
            
            fig = px.bar(
                x=hours,
                y=throughputs,
                labels={'x': 'Hour', 'y': 'Products Inspected'},
                color=throughputs,
                color_continuous_scale='Blues'
            )
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=250,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Recent Detections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Recent Detections")
        
        if use_real_data:
            # Use real detection data
            df_display = df.head(15)[['timestamp', 'image_id', 'decision', 'defect_type', 'confidence', 'latency_ms']].copy()
            df_display.columns = ['Time', 'Image ID', 'Decision', 'Defect Type', 'Confidence', 'Latency (ms)']
        else:
            # Generate sample detection data
            n_samples = 10
            detections_data = {
                'Time': pd.date_range(end=datetime.now(), periods=n_samples, freq='30S'),
                'Image ID': [f'IMG_{np.random.randint(10000, 99999)}' for _ in range(n_samples)],
                'Decision': np.random.choice(['ACCEPT', 'REJECT', 'ALERT'], size=n_samples, p=[0.85, 0.10, 0.05]),
                'Defect Type': np.random.choice(['None', 'Scratch', 'Crack', 'Dent', 'Contamination'], size=n_samples),
                'Confidence': np.random.uniform(0.7, 0.99, size=n_samples),
                'Latency (ms)': np.random.uniform(20, 80, size=n_samples)
            }
            df_display = pd.DataFrame(detections_data)
        
        # Style the dataframe
        def color_decision(val):
            if val == 'ACCEPT':
                return 'background-color: #d4edda'
            elif val == 'REJECT':
                return 'background-color: #f8d7da'
            elif val == 'ALERT':
                return 'background-color: #fff3cd'
            return ''
        
        styled_df = df_display.style.applymap(color_decision, subset=['Decision'])
        styled_df = styled_df.format({
            'Confidence': '{:.2%}',
            'Latency (ms)': '{:.1f}'
        })
        
        st.dataframe(styled_df, use_container_width=True, height=350)
    
    with col2:
        st.subheader("🚨 Active Alerts")
        
        alerts = [
            {"severity": "⚠️ WARNING", "message": "High defect rate on Line 2", "time": "2 min ago"},
            {"severity": "ℹ️ INFO", "message": "Model updated successfully", "time": "15 min ago"},
            {"severity": "🔴 ERROR", "message": "Camera 3 connection lost", "time": "1 hr ago"},
        ]
        
        for alert in alerts:
            with st.container():
                st.markdown(f"**{alert['severity']}**")
                st.markdown(f"{alert['message']}")
                st.caption(alert['time'])
                st.divider()
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col2:
        st.caption(f"Line {line_id} | {time_range}")
    
    with col3:
        st.caption("Smart Industrial Defect Detection v1.0")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    create_dashboard()
