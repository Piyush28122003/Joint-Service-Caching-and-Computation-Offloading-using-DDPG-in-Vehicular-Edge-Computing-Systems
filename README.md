# Vehicle Edge Computing (VEC) with DDPG

This project implements a Deep Deterministic Policy Gradient (DDPG) approach for optimizing task offloading and service caching in a Vehicle Edge Computing environment.

## Project Structure

```
.
├── ddpg_model/          # DDPG implementation
│   └── ddpg.py         # DDPG agent implementation
├── environment/         # Environment implementation
│   └── vec_env.py      # VEC environment
├── results/            # Trained models and metrics
├── main.py            # Main simulation script
├── train_model.py     # Training script
└── requirements.txt   # Project dependencies
```

## Environment Parameters

🚗 Vehicles
- Density: 2-5 vehicles per edge node
- Storage capacity: 50 MB
- CPU frequency: 5×10⁸ cycles/s

🖥️ Edge Servers
- Number of servers: 3
- Communication range: 500 meters
- Bandwidth: 20 MHz
- Storage capacity: 100 MB
- CPU frequency: 1×10⁹ cycles/s

🧠 Caching & Computation
- Service programs: 5
- Service program size: 50 MB
- Task input size: 20 MB
- Computation intensity: 10⁵ cycles/bit
- Edge-to-edge rate: 15 Mbps
- Edge-to-cloud rate: 10 Mbps
- Edge power: 1 W
- Cloud power: 2 W
- Energy efficiency: 1×10⁻²⁶

⏱️ Simulation
- Time slots: 40
- Slot duration: 30 seconds
- One task per vehicle per time slot

## DDPG Parameters

- Actor learning rate: 0.001
- Critic learning rate: 0.002
- Replay buffer size: 10,000
- Mini-batch size: 128
- Discount factor (γ): 0.99
- Soft update (τ): 0.01

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python train_model.py
   ```
   This will train the DDPG model and save it in the `results/trained_model` directory.

3. Run simulation:
   ```bash
   python main.py
   ```
   Enter the number of vehicles when prompted (2-15) to run the simulation.

## Results

The training process generates:
- Best model weights (`ddpg_best.pth`)
- Training metrics plot (`training_metrics.png`)
- Environment parameters (`env_params.json`)

The simulation shows:
- Real-time task execution details
- Final statistics including:
  - Task offloading decisions
  - Processing delays
  - Energy consumption 