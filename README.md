# README

## Overview

This repository contains the code necessary for running our GPU based mpQP solver. The code is specifically optimized for NVIDIA RTX 4090. ensure that CUDA 12.2 and Gurobi 10.0.3 are correctly installed along with the specified versions of additional dependencies.

## Installation

### Prerequisites

- **CUDA 12.2**: Ensure you have CUDA 12.2 installed. Follow the installation guide from the [official NVIDIA CUDA documentation](https://developer.nvidia.com/cuda-toolkit).
- **Gurobi 10.0.3**: You need to install Gurobi 10.0.3. Visit the [Gurobi installation guide](https://www.gurobi.com/documentation/) for detailed instructions.
- **Python**: Ensure you have Python installed (preferably Python 3.8 or higher).

### Dependencies

Install the required Python packages using pip:

```bash
pip install ppopt==1.5.1
pip install numpy==1.24.3
```

## Running the Code

To run the main application, execute the following command in your terminal:

```bash
python main.py
```

## Optimising for Different GPUs

This codebase has been optimized for different GPU architectures:

- **RTX 4090**: The provided `.so` files are optimized for RTX 4090.
- **H100**: Modify the CUDA code to use `size_t` instead of `int` to enhance performance on H100 GPUs.
