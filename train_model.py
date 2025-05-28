import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from environment.vec_env import VECEnvironment
from ddpg_model.ddpg import DDPG

def create_environment(num_vehicles=6):
    """Create and initialize the VEC environment."""
    # Vehicle parameters
    vehicle_params = {
        "storage_capacity": 50,  # MB
        "cpu_freq": 5e8,  # cycles/s
    }
    
    # Edge server parameters
    edge_params = {
        "range": 500,  # meters
        "bandwidth": 20,  # MHz
        "storage": 100,  # MB
        "cpu_freq": 1e9,  # cycles/s
    }
    
    # Task parameters
    task_params = {
        "num_services": 2,  # Number of services per vehicle
        "service_size": 50,  # MB
        "input_size": 20,  # MB
        "computation_intensity": 1e5,  # cycles/bit
    }
    
    # Network parameters
    network_params = {
        "edge_rate": 100,  # Mbps
        "cloud_rate": 50,  # Mbps
        "edge_power": 1,  # W
        "cloud_power": 2,  # W
        "energy_efficiency": 1e-26,  # κ constant
    }
    
    # Simulation parameters
    simulation_params = {
        "time_slots": 40,
        "slot_duration": 30,  # seconds
    }
    
    return VECEnvironment(
        num_vehicles=num_vehicles,
        num_edge_servers=3,
        vehicle_params=vehicle_params,
        edge_params=edge_params,
        task_params=task_params,
        network_params=network_params,
        simulation_params=simulation_params
    )

def plot_training_metrics(metrics, save_dir):
    """Plot training metrics similar to the paper figures."""
    try:
        # Figure 5: Total delay per episode
        plt.figure(figsize=(10, 6))
        plt.plot(metrics["episode_delays"], 'r-', label='DDPG-based edge caching and offloading')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Total delay per episode')
        plt.title('Performance of total delay per episode')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "delay_per_episode.png"))
        plt.close()
        
        # Figure 6: Total energy per episode
        plt.figure(figsize=(10, 6))
        plt.plot(metrics["episode_energy"], 'r-', label='DDPG-based edge caching and offloading')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Total energy per episode')
        plt.title('Performance of total energy consumption per episode')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "energy_per_episode.png"))
        plt.close()
        
        # Figure 7: Vehicle density effect
        vehicle_densities = [2, 3, 4, 5]
        delays_by_density = {}
        for d, delays in zip(vehicle_densities, metrics["density_delays"]):
            if delays:  # Only calculate mean if we have data
                delays_by_density[d] = np.nanmean(delays)  # Use nanmean to handle NaN values
            else:
                delays_by_density[d] = 0
        
        plt.figure(figsize=(10, 6))
        plt.bar(vehicle_densities, [delays_by_density.get(d, 0) for d in vehicle_densities],
                color='r', alpha=0.7, label='DDPG-based edge caching and offloading')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('The vehicle density at each edge')
        plt.ylabel('Total delay in the period')
        plt.title('Effect of vehicle density on total task processing delay')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "density_effect.png"))
        plt.close()
        
        # Figure 8: Edge caching decisions
        plt.figure(figsize=(12, 6))
        cache_stats = {
            "hits": metrics["cache_hits"],
            "misses": metrics["cache_misses"]
        }
        x = np.arange(len(cache_stats["hits"]))
        width = 0.35
        
        plt.bar(x - width/2, cache_stats["hits"], width, label='Cache Hits', color='g', alpha=0.7)
        plt.bar(x + width/2, cache_stats["misses"], width, label='Cache Misses', color='r', alpha=0.7)
        plt.xlabel('Edge caching decision cases')
        plt.ylabel('Number of cases')
        plt.title('Edge caching decision statistics')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(save_dir, "cache_decisions.png"))
        plt.close()
        
        # Figure 9: Task size vs Delay
        if metrics["task_sizes"] and metrics["processing_delays"]:  # Only plot if we have data
            plt.figure(figsize=(10, 6))
            task_sizes = np.array(metrics["task_sizes"])
            delays = np.array(metrics["processing_delays"])
            
            # Remove any NaN values
            valid_idx = ~np.isnan(task_sizes) & ~np.isnan(delays)
            task_sizes = task_sizes[valid_idx]
            delays = delays[valid_idx]
            
            if len(task_sizes) > 0:  # Only plot if we have valid data
                # Sort by task size for plotting
                sort_idx = np.argsort(task_sizes)
                task_sizes = task_sizes[sort_idx]
                delays = delays[sort_idx]
                
                plt.plot(task_sizes, delays, 'r-', label='DDPG-based edge caching and offloading')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xlabel('Task size')
                plt.ylabel('Total delay (unnormalized)')
                plt.title('Total delay versus task size')
                plt.legend()
                plt.savefig(os.path.join(save_dir, "size_vs_delay.png"))
            plt.close()
        
        # Additional metrics
        plt.figure(figsize=(15, 10))
        
        # Server utilization
        plt.subplot(2, 2, 1)
        for i, util in enumerate(metrics["server_utilization"]):
            if util:  # Only plot if we have data
                plt.plot(util, label=f'Server {i+1}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Utilization')
        plt.title('Edge Server Utilization')
        plt.legend()
        
        # Execution distribution
        plt.subplot(2, 2, 2)
        exec_data = [
            metrics["local_executions"][-1] if metrics["local_executions"] else 0,
            metrics["edge_executions"][-1] if metrics["edge_executions"] else 0,
            metrics["cloud_executions"][-1] if metrics["cloud_executions"] else 0
        ]
        if sum(exec_data) > 0:  # Only create pie chart if we have non-zero data
            plt.pie(exec_data, labels=['Local', 'Edge', 'Cloud'], autopct='%1.1f%%')
        plt.title('Task Execution Distribution')
        
        # Bandwidth usage
        plt.subplot(2, 2, 3)
        for i, bw in enumerate(metrics["bandwidth_usage"]):
            if bw:  # Only plot if we have data
                plt.plot(bw, label=f'Server {i+1}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Bandwidth (MB)')
        plt.title('Edge Server Bandwidth Usage')
        plt.legend()
        
        # Energy distribution
        plt.subplot(2, 2, 4)
        if metrics["energy_consumption"]:  # Only plot if we have data
            energy_data = np.array(metrics["energy_consumption"])
            valid_energy = energy_data[~np.isnan(energy_data)]  # Remove NaN values
            if len(valid_energy) > 0:
                plt.hist(valid_energy, bins=50, color='g', alpha=0.7)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Energy Consumption')
        plt.ylabel('Frequency')
        plt.title('Energy Consumption Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "additional_metrics.png"))
        plt.close()
    
    except Exception as e:
        print(f"Warning: Error while plotting metrics: {str(e)}")
        plt.close('all')  # Clean up any open figures

def train(env, model, num_episodes=3000, max_steps=100, save_interval=100):
    """Train the DDPG agent."""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save environment parameters
    env_params = {
        "state_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.shape[0]
    }
    with open(os.path.join(results_dir, "env_params.json"), "w") as f:
        json.dump(env_params, f)
    
    # Training metrics
    metrics = {
        "episode_rewards": [],
        "episode_delays": [],
        "episode_energy": [],
        "density_delays": [[] for _ in range(4)],  # For 2-5 vehicles per edge
        "cache_hits": [],
        "cache_misses": [],
        "task_sizes": [],
        "processing_delays": [],
        "server_utilization": [[] for _ in range(env.num_edge_servers)],
        "bandwidth_usage": [[] for _ in range(env.num_edge_servers)],
        "local_executions": [],
        "edge_executions": [],
        "cloud_executions": [],
        "energy_consumption": []
    }
    
    best_reward = float("-inf")
    
    # Training loop
    progress_bar = tqdm(range(num_episodes), desc="Training")
    for episode in progress_bar:
        state, _ = env.reset()
        episode_reward = 0
        episode_delay = 0
        episode_energy = 0
        num_tasks = 0
        
        for step in range(max_steps):
            # Select action with noise for exploration
            action = model.select_action(state, add_noise=True)
            
            # Execute action
            next_state, reward, done, _, info = env.step(action)
            
            # Store experience
            model.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train model
            model.train()
            
            # Update metrics
            episode_reward += reward
            episode_delay += info["total_delay"]
            episode_energy += info["total_energy"]
            num_tasks += info["num_tasks"]
            
            # Update detailed metrics
            for i in range(env.num_edge_servers):
                metrics["server_utilization"][i].append(info["server_utilization"][i])
                metrics["bandwidth_usage"][i].append(info["bandwidth_usage"][i])
            
            if done:
                break
                
            state = next_state
        
        # Record episode metrics
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_delays"].append(episode_delay / max(1, num_tasks))
        metrics["episode_energy"].append(episode_energy)
        
        # Record density-specific delays
        vehicle_density = len(env.vehicles) // env.num_edge_servers
        if 2 <= vehicle_density <= 5:
            metrics["density_delays"][vehicle_density-2].append(episode_delay / max(1, num_tasks))
        
        # Record execution statistics
        metrics["local_executions"].append(info["local_executions"])
        metrics["edge_executions"].append(info["edge_executions"])
        metrics["cloud_executions"].append(info["cloud_executions"])
        
        # Record cache statistics
        metrics["cache_hits"].append(info["edge_cache_hits"])
        metrics["cache_misses"].append(info["edge_cache_misses"])
        
        # Extend task-specific metrics
        metrics["task_sizes"].extend(env.metrics["task_sizes"])
        metrics["processing_delays"].extend(env.metrics["processing_delays"])
        metrics["energy_consumption"].extend(env.metrics["energy_consumption"])
        
        # Update progress bar
        progress_bar.set_postfix({
            "reward": f"{episode_reward:.2f}",
            "delay": f"{metrics['episode_delays'][-1]:.2f}s",
            "energy": f"{episode_energy:.2f}J"
        })
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            model.save(os.path.join(results_dir, "ddpg_best.pth"))
        
        # Save intermediate results and plots
        if (episode + 1) % save_interval == 0:
            model.save(os.path.join(results_dir, f"ddpg_episode_{episode+1}.pth"))
            plot_training_metrics(metrics, results_dir)

def main():
    # Create environment
    env = create_environment()
    
    # Initialize DDPG agent
    model = DDPG(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        lr_actor=1e-4,
        lr_critic=1e-3
    )
    
    # Train agent
    train(env, model)

if __name__ == "__main__":
    main() 