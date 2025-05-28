import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
from glob import glob
import os
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="VEC Simulation Dashboard",
    page_icon="🚗",
    layout="wide"
)

def load_report_data(report_dir):
    """Load report data from the specified directory."""
    with open(os.path.join(report_dir, "execution_report.json")) as f:
        return json.load(f)

def create_offloading_pie_chart(report_data):
    """Create a pie chart of offloading distribution."""
    offload_dist = report_data["summary_statistics"]["offloading_distribution"]
    fig = go.Figure(data=[go.Pie(
        labels=["Local", "Edge", "Cloud"],
        values=[offload_dist["local"], offload_dist["edge"], offload_dist["cloud"]],
        hole=.3
    )])
    fig.update_layout(title="Task Offloading Distribution")
    return fig

def format_report_name(report_path):
    """Format the report name from the path."""
    return os.path.basename(report_path)

def main():
    st.title("🚗 VEC Simulation Dashboard")
    
    # Sidebar for controls
    st.sidebar.title("Controls")
    report_dirs = glob("results/report_*")
    
    if not report_dirs:
        st.error("No simulation reports found. Please run the simulation first!")
        return
    
    selected_report = st.sidebar.selectbox(
        "Select Report",
        report_dirs,
        format_func=format_report_name
    )
    
    try:
        # Load report data
        report_data = load_report_data(selected_report)
        
        # Display summary metrics in a nice grid
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Tasks",
                report_data["summary_statistics"]["total_tasks"],
                delta=None
            )
        with col2:
            st.metric(
                "Avg Processing Time",
                f"{report_data['summary_statistics']['average_delay']:.2f}s",
                delta=None
            )
        with col3:
            st.metric(
                "Avg Energy/Task",
                f"{report_data['summary_statistics']['average_energy']:.2f}J",
                delta=None
            )
        with col4:
            st.metric(
                "Total Energy",
                f"{report_data['summary_statistics']['total_energy']:.2f}J",
                delta=None
            )
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Performance", "⚡ Energy", "💾 Cache"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(f"{selected_report}/task_processing_dist.png"):
                    st.image(f"{selected_report}/task_processing_dist.png")
                else:
                    st.warning("Task processing distribution plot not available")
            with col2:
                st.plotly_chart(create_offloading_pie_chart(report_data), use_container_width=True)
            if os.path.exists(f"{selected_report}/performance_trends.png"):
                st.image(f"{selected_report}/performance_trends.png")
            else:
                st.warning("Performance trends plot not available")
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(f"{selected_report}/energy_pattern.png"):
                    st.image(f"{selected_report}/energy_pattern.png")
                else:
                    st.warning("Energy pattern plot not available")
            with col2:
                try:
                    energy_metrics = report_data["performance_metrics"]["energy"]
                    fig = go.Figure()
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=report_data["summary_statistics"]["average_energy"],
                        title={'text': "Average Energy Consumption (J)"},
                        gauge={'axis': {'range': [energy_metrics['min'], energy_metrics['max']]}}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                except KeyError:
                    st.warning("Energy metrics not available")
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(f"{selected_report}/cache_performance.png"):
                    st.image(f"{selected_report}/cache_performance.png")
                else:
                    st.warning("Cache performance plot not available")
            with col2:
                if os.path.exists(f"{selected_report}/vehicle_loads.png"):
                    st.image(f"{selected_report}/vehicle_loads.png")
                else:
                    st.warning("Vehicle loads plot not available")
        
        # Display detailed statistics
        with st.expander("📈 Detailed Statistics"):
            st.json(report_data)
    
    except Exception as e:
        st.error(f"Error loading report data: {str(e)}")
        st.error("Please make sure the report files are in the correct format.")

if __name__ == "__main__":
    main() 