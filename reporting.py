import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from datetime import datetime
import os
import numpy as np

class VECReporter:
    def __init__(self, output_dir="results"):
        """Initialize the reporter with an output directory."""
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = os.path.join(output_dir, f"report_{self.timestamp}")
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            "tasks": [],
            "delays": [],
            "energy": [],
            "offloading": [],
            "cache": [],
            "server_util": []
        }
    
    def add_task(self, task_id, vehicle_id, size, delay, energy, offload_ratios):
        """Add a task execution record."""
        self.metrics["tasks"].append({
            "task_id": task_id,
            "vehicle_id": vehicle_id,
            "size": size,
            "delay": delay,
            "energy": energy,
            "local_ratio": offload_ratios[0],
            "edge_ratio": offload_ratios[1],
            "cloud_ratio": offload_ratios[2]
        })
    
    def update_server_utilization(self, server_id, utilization, bandwidth):
        """Update server utilization metrics."""
        self.metrics["server_util"].append({
            "server_id": server_id,
            "utilization": utilization,
            "bandwidth": bandwidth
        })
    
    def update_cache_metrics(self, hits, misses):
        """Update cache performance metrics."""
        self.metrics["cache"].append({
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / (hits + misses) if (hits + misses) > 0 else 0
        })
    
    def generate_graphs(self):
        """Generate all visualization graphs."""
        # Convert tasks data to DataFrame
        df_tasks = pd.DataFrame(self.metrics["tasks"])
        
        # 1. Task Processing Time Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df_tasks, x="delay", bins=30)
        plt.title("Distribution of Task Processing Times")
        plt.xlabel("Processing Time (s)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(self.report_dir, "task_processing_dist.png"))
        plt.close()
        
        # 2. Energy Consumption Pattern
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_tasks, x="size", y="energy")
        plt.title("Energy Consumption vs Task Size")
        plt.xlabel("Task Size (MB)")
        plt.ylabel("Energy Consumption (J)")
        plt.savefig(os.path.join(self.report_dir, "energy_pattern.png"))
        plt.close()
        
        # 3. Offloading Distribution Over Time
        plt.figure(figsize=(12, 6))
        df_tasks[["local_ratio", "edge_ratio", "cloud_ratio"]].rolling(window=20).mean().plot()
        plt.title("Offloading Distribution Trends")
        plt.xlabel("Task Number")
        plt.ylabel("Ratio")
        plt.legend(["Local", "Edge", "Cloud"])
        plt.savefig(os.path.join(self.report_dir, "offloading_trends.png"))
        plt.close()
        
        # 4. Vehicle Load Distribution
        plt.figure(figsize=(10, 6))
        vehicle_loads = df_tasks.groupby("vehicle_id").size()
        sns.barplot(x=vehicle_loads.index, y=vehicle_loads.values)
        plt.title("Task Distribution Across Vehicles")
        plt.xlabel("Vehicle ID")
        plt.ylabel("Number of Tasks")
        plt.savefig(os.path.join(self.report_dir, "vehicle_loads.png"))
        plt.close()
        
        # 5. Performance Metrics Over Time
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Processing Time Trend
        df_tasks["delay"].rolling(window=20).mean().plot(ax=ax1)
        ax1.set_title("Average Processing Time Trend")
        ax1.set_xlabel("Task Number")
        ax1.set_ylabel("Processing Time (s)")
        
        # Energy Consumption Trend
        df_tasks["energy"].rolling(window=20).mean().plot(ax=ax2)
        ax2.set_title("Average Energy Consumption Trend")
        ax2.set_xlabel("Task Number")
        ax2.set_ylabel("Energy (J)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.report_dir, "performance_trends.png"))
        plt.close()
        
        # 6. Cache Performance
        if self.metrics["cache"]:
            df_cache = pd.DataFrame(self.metrics["cache"])
            plt.figure(figsize=(10, 6))
            df_cache["hit_rate"].plot()
            plt.title("Cache Hit Rate Over Time")
            plt.xlabel("Time")
            plt.ylabel("Hit Rate")
            plt.savefig(os.path.join(self.report_dir, "cache_performance.png"))
            plt.close()
    
    def generate_report(self):
        """Generate a detailed execution report."""
        df_tasks = pd.DataFrame(self.metrics["tasks"])
        
        report = {
            "execution_timestamp": self.timestamp,
            "summary_statistics": {
                "total_tasks": len(df_tasks),
                "average_delay": df_tasks["delay"].mean(),
                "average_energy": df_tasks["energy"].mean(),
                "total_energy": df_tasks["energy"].sum(),
                "offloading_distribution": {
                    "local": df_tasks["local_ratio"].mean(),
                    "edge": df_tasks["edge_ratio"].mean(),
                    "cloud": df_tasks["cloud_ratio"].mean()
                }
            },
            "performance_metrics": {
                "delay": {
                    "min": df_tasks["delay"].min(),
                    "max": df_tasks["delay"].max(),
                    "std": df_tasks["delay"].std()
                },
                "energy": {
                    "min": df_tasks["energy"].min(),
                    "max": df_tasks["energy"].max(),
                    "std": df_tasks["energy"].std()
                }
            }
        }
        
        # Save report as JSON
        with open(os.path.join(self.report_dir, "execution_report.json"), "w") as f:
            json.dump(report, f, indent=4)
        
        # Generate a human-readable summary
        summary = f"""
VEC Simulation Execution Report
==============================
Timestamp: {self.timestamp}

Task Statistics
--------------
Total Tasks Processed: {report['summary_statistics']['total_tasks']}
Average Processing Time: {report['summary_statistics']['average_delay']:.3f} s
Average Energy Consumption: {report['summary_statistics']['average_energy']:.3f} J
Total Energy Consumption: {report['summary_statistics']['total_energy']:.3f} J

Offloading Distribution
----------------------
Local Execution: {report['summary_statistics']['offloading_distribution']['local']:.2%}
Edge Execution:  {report['summary_statistics']['offloading_distribution']['edge']:.2%}
Cloud Execution: {report['summary_statistics']['offloading_distribution']['cloud']:.2%}

Performance Metrics
------------------
Processing Time (s):
  - Min: {report['performance_metrics']['delay']['min']:.3f}
  - Max: {report['performance_metrics']['delay']['max']:.3f}
  - Std: {report['performance_metrics']['delay']['std']:.3f}

Energy Consumption (J):
  - Min: {report['performance_metrics']['energy']['min']:.3f}
  - Max: {report['performance_metrics']['energy']['max']:.3f}
  - Std: {report['performance_metrics']['energy']['std']:.3f}
"""
        
        # Save human-readable summary
        with open(os.path.join(self.report_dir, "summary.txt"), "w") as f:
            f.write(summary)
        
        return report 