import os
import json
from train_model import create_environment
from ddpg_model.ddpg import DDPG
import numpy as np
from typing import Dict, List
import time
from datetime import datetime
from reporting import VECReporter

def load_model(model_path: str) -> tuple[DDPG, Dict]:
    """Load trained DDPG model and environment parameters."""
    # Load environment parameters
    with open(os.path.join(model_path, "env_params.json"), "r") as f:
        env_params = json.load(f)
    
    # Create and load model
    model = DDPG(
        state_dim=env_params["state_dim"],
        action_dim=env_params["action_dim"]
    )
    model.load(os.path.join(model_path, "ddpg_best.pth"))
    
    return model, env_params

def get_offload_ratios(action: np.ndarray, vehicle_id: int, num_services: int) -> List[float]:
    """Extract offloading ratios for a specific vehicle."""
    # Calculate indices for the vehicle's actions
    start_idx = num_services + (vehicle_id * num_services)
    end_idx = start_idx + num_services
    
    # Get vehicle's offloading ratios
    oV = action[start_idx:end_idx]  # Edge offload ratios
    oE = action[-num_services:]  # Edge pool ratios
    
    # Calculate actual ratios
    edge_ratio = oV * (1.0 - oE)
    cloud_ratio = oV * oE
    local_ratio = 1.0 - oV
    
    return [local_ratio.mean(), edge_ratio.mean(), cloud_ratio.mean()]

def print_task_details(task_id: int, vehicle_id: int, size: float, delay: float, energy: float, offload_ratios: List[float]):
    """Print details of a task execution."""
    print("\n" + "="*50)
    print(f"🚗 Task {task_id:3d} | Vehicle {vehicle_id:2d}")
    print("-"*50)
    print(f"📦 Task Size:        {size:6.2f} MB")
    print(f"⏱️  Processing Time:  {delay:6.3f} s")
    print(f"⚡ Energy Used:      {energy:6.3f} J")
    print("\n📊 Offloading Distribution:")
    print(f"   • Local Execution: {offload_ratios[0]:6.2%}")
    print(f"   • Edge Execution:  {offload_ratios[1]:6.2%}")
    print(f"   • Edge Pool:       {offload_ratios[2]:6.2%}")
    print("="*50)

def main():
    # Ask user for number of vehicles
    while True:
        try:
            num_vehicles = int(input("Enter number of vehicles (5-20): "))
            if 5 <= num_vehicles <= 20:
                break
            print("Please enter a number between 5 and 20")
        except ValueError:
            print("Please enter a valid number")
    
    # Create environment with specified number of vehicles
    env = create_environment(num_vehicles=num_vehicles)
    
    # Load pre-trained model
    model, _ = load_model("results")
    
    # Initialize metrics
    total_delay = 0.0
    total_energy = 0.0
    num_tasks = 0
    task_id = 0
    cache_hits = 0
    cache_misses = 0
    local_execs = 0
    edge_execs = 0
    cloud_execs = 0
    server_utils = [0.0] * env.num_edge_servers
    
    # Initialize reporter
    reporter = VECReporter()
    
    # Reset environment
    state, _ = env.reset()
    
    print("\n🚀 Starting VEC Simulation")
    print("="*50)
    print(f"Number of Vehicles: {num_vehicles}")
    print(f"Number of Edge Servers: {env.num_edge_servers}")
    print("="*50)
    
    try:
        while True:
            # Get model's action with exploration noise for variety
            action = model.select_action(state, add_noise=True)
            
            # Execute action
            next_state, reward, done, _, info = env.step(action)
            
            # Update metrics
            total_delay += info["total_delay"]
            total_energy += info["total_energy"]
            num_tasks += info["num_tasks"]
            cache_hits += info["edge_cache_hits"]
            cache_misses += info["edge_cache_misses"]
            local_execs += info["local_executions"]
            edge_execs += info["edge_executions"]
            cloud_execs += info["cloud_executions"]
            
            # Update server utilization
            for i, util in enumerate(info["server_utilization"]):
                server_utils[i] = max(server_utils[i], util)
                reporter.update_server_utilization(i, util, info["bandwidth_usage"][i])
            
            # Update cache metrics
            reporter.update_cache_metrics(info["edge_cache_hits"], info["edge_cache_misses"])
            
            # Print task details
            if info["num_tasks"] > 0:
                for _ in range(info["num_tasks"]):
                    task_id += 1
                    vehicle_id = task_id % num_vehicles
                    # Get actual offloading ratios from model's action
                    offload_ratios = get_offload_ratios(action, vehicle_id, env.num_services)
                    
                    # Print task details
                    print_task_details(
                        task_id,
                        vehicle_id,
                        env.task_params["input_size"],
                        info["total_delay"] / info["num_tasks"],
                        info["total_energy"] / info["num_tasks"],
                        offload_ratios
                    )
                    
                    # Add task to reporter
                    reporter.add_task(
                        task_id,
                        vehicle_id,
                        env.task_params["input_size"],
                        info["total_delay"] / info["num_tasks"],
                        info["total_energy"] / info["num_tasks"],
                        offload_ratios
                    )
            
            if done:
                break
            
            state = next_state
            time.sleep(0.5)  # Slow down output for readability
    
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    
    finally:
        # Generate graphs and reports
        print("\n📊 Generating performance reports and visualizations...")
        reporter.generate_graphs()
        report = reporter.generate_report()
        
        print(f"\n✅ Reports and visualizations saved to: {reporter.report_dir}")
        print("\nSimulation completed!")

if __name__ == "__main__":
    main() 