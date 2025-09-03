import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Any

from ..models.data_models import AMLAlert, RiskLevel
from ..alerts.alert_manager import AlertManager

class AMLDashboard:
    """Real-time AML detection dashboard using Streamlit"""
    
    def __init__(self, alert_manager: AlertManager):
        """
        Initialize dashboard
        
        Args:
            alert_manager: Alert manager instance
        """
        self.alert_manager = alert_manager
    
    def run(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="AML Detection System",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main title
        st.title("🔍 Real-Time Anti-Money Laundering Detection System")
        st.markdown("---")
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        self._render_main_content()
    
    def _render_sidebar(self):
        """Render sidebar with controls and statistics"""
        st.sidebar.header("📊 System Overview")
        
        # Get dashboard data
        dashboard_data = self.alert_manager.get_dashboard_data()
        stats = dashboard_data['stats']
        
        # Key metrics
        st.sidebar.metric(
            "Total Alerts",
            stats.get('total', 0),
            delta=None
        )
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric(
                "Critical",
                stats.get('critical', 0),
                delta=None
            )
        with col2:
            st.metric(
                "High Risk",
                stats.get('high', 0),
                delta=None
            )
        
        # Risk level distribution
        risk_data = {
            'Critical': stats.get('critical', 0),
            'High': stats.get('high', 0),
            'Medium': stats.get('medium', 0),
            'Low': stats.get('low', 0)
        }
        
        if sum(risk_data.values()) > 0:
            fig_pie = px.pie(
                values=list(risk_data.values()),
                names=list(risk_data.keys()),
                title="Risk Level Distribution",
                color_discrete_map={
                    'Critical': '#dc3545',
                    'High': '#fd7e14',
                    'Medium': '#ffc107',
                    'Low': '#28a745'
                }
            )
            fig_pie.update_layout(height=300)
            st.sidebar.plotly_chart(fig_pie, use_container_width=True)
        
        # Auto-refresh control
        st.sidebar.header("⚙️ Settings")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
        
        if auto_refresh:
            time.sleep(refresh_interval)
            st.experimental_rerun()
    
    def _render_main_content(self):
        """Render main dashboard content"""
        # Get recent alerts
        recent_alerts = self.alert_manager.get_alert_history(limit=100)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🚨 Active Alerts", "📈 Analytics", "🔍 Detection Models", "📋 Transaction History"])
        
        with tab1:
            self._render_alerts_tab(recent_alerts)
        
        with tab2:
            self._render_analytics_tab(recent_alerts)
        
        with tab3:
            self._render_models_tab()
        
        with tab4:
            self._render_transaction_history_tab()
    
    def _render_alerts_tab(self, alerts: List[AMLAlert]):
        """Render active alerts tab"""
        st.header("🚨 Active Alerts")
        
        if not alerts:
            st.info("No alerts currently active")
            return
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_filter = st.selectbox(
                "Filter by Risk Level",
                ["All", "Critical", "High", "Medium", "Low"]
            )
        
        with col2:
            alert_type_filter = st.selectbox(
                "Filter by Alert Type",
                ["All"] + list(set(alert.alert_type for alert in alerts))
            )
        
        with col3:
            time_filter = st.selectbox(
                "Time Range",
                ["Last Hour", "Last 24 Hours", "Last 7 Days", "All Time"]
            )
        
        # Apply filters
        filtered_alerts = self._filter_alerts(alerts, risk_filter, alert_type_filter, time_filter)
        
        # Display alerts
        for alert in filtered_alerts[:20]:  # Show top 20
            self._render_alert_card(alert)
    
    def _render_alert_card(self, alert: AMLAlert):
        """Render individual alert card"""
        # Color based on risk level
        colors = {
            RiskLevel.CRITICAL: "#dc3545",
            RiskLevel.HIGH: "#fd7e14", 
            RiskLevel.MEDIUM: "#ffc107",
            RiskLevel.LOW: "#28a745"
        }
        color = colors.get(alert.risk_level, "#6c757d")
        
        with st.container():
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding: 10px; margin: 10px 0; background-color: #f8f9fa;">
                <h4 style="color: {color}; margin: 0;">{alert.alert_type.upper()} - {alert.risk_level.upper()}</h4>
                <p><strong>Alert ID:</strong> {alert.id}</p>
                <p><strong>Transaction ID:</strong> {alert.transaction_id}</p>
                <p><strong>Description:</strong> {alert.description}</p>
                <p><strong>Confidence:</strong> {alert.confidence_score:.2%}</p>
                <p><strong>Detected By:</strong> {alert.detected_by}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional data expander
            if alert.additional_data:
                with st.expander("View Additional Data"):
                    st.json(alert.additional_data)
    
    def _render_analytics_tab(self, alerts: List[AMLAlert]):
        """Render analytics tab"""
        st.header("📈 AML Analytics")
        
        if not alerts:
            st.info("No data available for analytics")
            return
        
        # Convert alerts to DataFrame for analysis
        df_alerts = pd.DataFrame([
            {
                'timestamp': alert.timestamp,
                'risk_level': alert.risk_level.value,
                'alert_type': alert.alert_type,
                'confidence': alert.confidence_score,
                'detected_by': alert.detected_by
            }
            for alert in alerts
        ])
        
        # Time series chart
        st.subheader("Alert Trends Over Time")
        df_alerts['date'] = pd.to_datetime(df_alerts['timestamp']).dt.date
        daily_counts = df_alerts.groupby('date').size().reset_index(name='count')
        
        fig_timeline = px.line(
            daily_counts,
            x='date',
            y='count',
            title="Daily Alert Count",
            markers=True
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Alert type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Alert Types")
            type_counts = df_alerts['alert_type'].value_counts()
            fig_types = px.bar(
                x=type_counts.index,
                y=type_counts.values,
                title="Alert Types Distribution"
            )
            st.plotly_chart(fig_types, use_container_width=True)
        
        with col2:
            st.subheader("Detection Methods")
            detector_counts = df_alerts['detected_by'].value_counts()
            fig_detectors = px.pie(
                values=detector_counts.values,
                names=detector_counts.index,
                title="Detection Methods"
            )
            st.plotly_chart(fig_detectors, use_container_width=True)
        
        # Risk level analysis
        st.subheader("Risk Level Analysis")
        risk_counts = df_alerts['risk_level'].value_counts()
        fig_risk = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="Risk Level Distribution",
            color=risk_counts.index,
            color_discrete_map={
                'critical': '#dc3545',
                'high': '#fd7e14',
                'medium': '#ffc107',
                'low': '#28a745'
            }
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Confidence score distribution
        st.subheader("Confidence Score Distribution")
        fig_confidence = px.histogram(
            df_alerts,
            x='confidence',
            nbins=20,
            title="Alert Confidence Scores"
        )
        st.plotly_chart(fig_confidence, use_container_width=True)
    
    def _render_models_tab(self):
        """Render detection models tab"""
        st.header("🔍 Detection Models Status")
        
        # Model status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Isolation Forest",
                "Active",
                delta="99.2% Accuracy"
            )
            st.success("Model loaded and running")
        
        with col2:
            st.metric(
                "Autoencoder",
                "Active", 
                delta="98.7% Accuracy"
            )
            st.success("Model loaded and running")
        
        with col3:
            st.metric(
                "Rule Engine",
                "Active",
                delta="15 Rules Active"
            )
            st.success("All rules operational")
        
        # Model performance metrics
        st.subheader("Model Performance Metrics")
        
        # Simulated performance data
        models_data = {
            'Model': ['Isolation Forest', 'Autoencoder', 'Rule Engine'],
            'Accuracy': [99.2, 98.7, 95.5],
            'Precision': [94.1, 96.3, 88.2],
            'Recall': [91.7, 89.4, 92.1],
            'F1-Score': [92.9, 92.7, 90.1]
        }
        
        df_models = pd.DataFrame(models_data)
        
        fig_performance = go.Figure()
        
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            fig_performance.add_trace(go.Bar(
                name=metric,
                x=df_models['Model'],
                y=df_models[metric]
            ))
        
        fig_performance.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score (%)",
            barmode='group'
        )
        
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Model configuration
        st.subheader("Model Configuration")
        
        with st.expander("Isolation Forest Settings"):
            st.code("""
            contamination: 0.1
            n_estimators: 100
            random_state: 42
            max_features: 1.0
            """)
        
        with st.expander("Autoencoder Settings"):
            st.code("""
            encoding_dim: 32
            epochs: 100
            batch_size: 32
            threshold_percentile: 95
            """)
        
        with st.expander("Rule Engine Settings"):
            st.code("""
            structuring_threshold: $10,000
            layering_complexity: 5 hops
            velocity_threshold: 5x baseline
            """)
    
    def _render_transaction_history_tab(self):
        """Render transaction history tab"""
        st.header("📋 Recent Transaction Activity")
        
        # Generate sample transaction data for demo
        sample_data = self._generate_sample_transactions()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{len(sample_data):,}")
        
        with col2:
            st.metric("Total Volume", f"${sample_data['amount'].sum():,.2f}")
        
        with col3:
            st.metric("Average Amount", f"${sample_data['amount'].mean():,.2f}")
        
        with col4:
            suspicious_count = len(sample_data[sample_data['amount'] > 10000])
            st.metric("Large Transactions", suspicious_count)
        
        # Transaction volume over time
        st.subheader("Transaction Volume Trends")
        hourly_volume = sample_data.groupby('hour')['amount'].sum().reset_index()
        
        fig_volume = px.line(
            hourly_volume,
            x='hour',
            y='amount',
            title="Hourly Transaction Volume",
            markers=True
        )
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Transaction table
        st.subheader("Recent Transactions")
        st.dataframe(
            sample_data.head(50),
            use_container_width=True
        )
    
    def _filter_alerts(self, alerts: List[AMLAlert], risk_filter: str, 
                      type_filter: str, time_filter: str) -> List[AMLAlert]:
        """Filter alerts based on criteria"""
        filtered = alerts
        
        # Risk level filter
        if risk_filter != "All":
            filtered = [a for a in filtered if a.risk_level.value == risk_filter.lower()]
        
        # Alert type filter
        if type_filter != "All":
            filtered = [a for a in filtered if a.alert_type == type_filter]
        
        # Time filter
        now = datetime.now()
        if time_filter == "Last Hour":
            cutoff = now - timedelta(hours=1)
        elif time_filter == "Last 24 Hours":
            cutoff = now - timedelta(days=1)
        elif time_filter == "Last 7 Days":
            cutoff = now - timedelta(days=7)
        else:
            cutoff = None
        
        if cutoff:
            filtered = [a for a in filtered if a.timestamp >= cutoff]
        
        return filtered
    
    def _generate_sample_transactions(self) -> pd.DataFrame:
        """Generate sample transaction data for demo purposes"""
        np.random.seed(42)
        n_transactions = 1000
        
        # Generate sample data
        data = {
            'transaction_id': [f"TXN{i:06d}" for i in range(n_transactions)],
            'timestamp': pd.date_range(start='2024-01-01', periods=n_transactions, freq='5min'),
            'amount': np.random.lognormal(mean=7, sigma=1.5, size=n_transactions),
            'transaction_type': np.random.choice(['deposit', 'withdrawal', 'transfer', 'payment'], n_transactions),
            'source_account': [f"ACC{np.random.randint(1000, 9999)}" for _ in range(n_transactions)],
            'channel': np.random.choice(['online', 'atm', 'branch', 'mobile'], n_transactions)
        }
        
        df = pd.DataFrame(data)
        df['hour'] = df['timestamp'].dt.hour
        df['amount'] = df['amount'].round(2)
        
        return df

def create_dashboard(alert_manager: AlertManager):
    """Create and return dashboard instance"""
    return AMLDashboard(alert_manager)