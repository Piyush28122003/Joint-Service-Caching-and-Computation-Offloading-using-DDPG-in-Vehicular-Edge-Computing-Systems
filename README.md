# 🚗 Joint Service Caching and Computation Offloading using DDPG in Vehicular Edge Computing Systems


## Overview
VEC-Optimizer is an advanced simulation and optimization framework for Vehicle Edge Computing (VEC) environments. It implements intelligent task offloading strategies using Deep Deterministic Policy Gradient (DDPG) to optimize processing time, energy consumption, and resource utilization in vehicular networks.

## Features
- **Intelligent Task Offloading**: Uses DDPG-based reinforcement learning for optimal task distribution
- **Multi-objective Optimization**: Balances processing time, energy consumption, and resource utilization
- **Real-time Visualization**: Interactive dashboard for monitoring system performance
- **Flexible Architecture**: Supports various vehicle configurations and task scenarios
- **Performance Analytics**: Comprehensive metrics and visualization tools

## Project Structure
```
VEC-Optimizer/
├── actor_critic_model/     # DDPG implementation
├── environment/            # VEC simulation environment
├── utils/                 # Utility functions and metrics
├── dashboard.py           # Interactive visualization dashboard
├── main.py               # Main simulation runner
└── requirements.txt      # Project dependencies
```

## Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Simulation
```bash
python main.py
```

### Viewing Results Dashboard
```bash
streamlit run dashboard.py
```

## Dashboard Features
- Real-time performance monitoring
- Task distribution visualization
- Energy consumption analysis
- Cache performance metrics
- Vehicle load distribution

## Performance Metrics
- Task Processing Time
- Energy Consumption
- Resource Utilization
- Offloading Distribution
  - Local Execution
  - Edge Server Execution
  - Edge Pool Distribution

## Requirements
- Python 3.8+
- PyTorch
- Streamlit
- Plotly
- Additional dependencies in requirements.txt

## Results
The system demonstrates effective task offloading with:
- Optimized processing times
- Reduced energy consumption
- Balanced resource utilization
- Adaptive task distribution based on network conditions

## Contributing
Contributions are welcome! Please feel free to submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

