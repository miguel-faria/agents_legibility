# Legible Agents using Deep RL

This repository focuses on developing legibility for agents using deep RL tools.

Legibility is an important ability for agents that have to be part of a team or have to collaborate with others in an 
environment.

Our work here focus on developing approaches that allow artificial agents to display legible behaviours both in pure 
collaborative scenarios -- scenarios where all members of a team have to cooperate to fulfill each objective/task -- or
mixed collaboration scenarios -- scenarios where some objectives/tasks require all agents to fulfill while others can be
fulfilled by a subset of team members.

## Installation

We provide three installation scripts.

- *install_miniconda.sh* - is used to install miniconda if you don't have it installed

  > source \<path-to-dir\>/install_miniconda.sh [\<path-install-miniconda\>] 

- *install_python_jax_env.sh* - used to install a python virtual environment (either using conda or venv+pip) that uses JAX for CUDA support

  > source \<path-to-dir\>/install_python_jax_env.sh -n \<env-name\> -t \<conda|pip\> [--llm 1|0] [-p \<python-version\>] [-c \<path-conda-home\>]

- *install_python_pytorch_env.sh* - used to install a python virtual environment (either using conda or venv+pip) that uses Pytorch for CUDA support

  > source \<path-to-dir\>/install_python_pytorch_env.sh -n \<env-name\> -t \<conda|pip\> [--llm 1|0] [-p \<python-version\>] [-c \<path-conda-home\>]
 
## Run

In the *scripts* folder there are scripts for both training and testing models for the lb-foraging and pursuit-evasion 
scenarios. These scripts also support slurm usage, just need to adapt the sbatch parameters for your cluster configurations.

