# Criticality Awareness QLS

This repository contains implementations and analysis of the Hierarchical Ferromagnetic Model (HFM) and Layer Dynamics Model for studying critical phenomena in neural systems.

## Overview

The project explores the relationship between internal feature organization and emergent learning behavior through:

1. **Hierarchical Ferromagnetic Model (HFM)**: A statistical physics model that exhibits phase transitions
2. **Layer Dynamics Model**: A system where each layer integrates its input over time with memory parameter ε

## Features

- Implementation of HFM with configurable parameters
- Calculation and visualization of specific heat C(g) to identify phase transitions
- Probability distribution analysis for different coupling strengths
- Support for various system sizes (n) and coupling parameters (g)

## Installation

```bash
# Clone the repository
git clone https://github.com/JS-Rojas/Criticality_Awareness_QLS.git
cd Criticality_Awareness_QLS

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The main analysis is contained in the Jupyter notebook `HFM_model.ipynb`. To run it:

```bash
jupyter notebook HFM_model.ipynb
```

You can also use the Python module directly:

```python
from hfm_distribution import plot_specific_heat, plot_HFM_distribution
import numpy as np

# Plot specific heat for different system sizes
g_range = np.linspace(0.0, 1.5, 100)
n_values = [10, 20, 30]
plot_specific_heat(n_values, g_range)
```

## Mathematical Background

The HFM model exhibits a phase transition at gc = log(2), which is signaled by a divergence in the specific heat:

C(g) = E[(H-E[H])²]

where H is the system's Hamiltonian and E[H] is its expected value.

## Author

Juan S. Rojas - [@JS-Rojas](https://github.com/JS-Rojas)

## License

MIT License - see LICENSE file for details 