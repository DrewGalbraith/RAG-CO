
# Team CO Repository 📝

This project implements and evaluates RAG pipelines.


## Installation and Setup

To get started with this project, first clone this repository using the following command:

```bash
git clone git@ghe.arlis.umd.edu:RISC/RISC-2024-CO.git
cd RISC-2024-CO
```

### 1: Development Environment

Ensure you have Python 3.11 installed. If you do not have Python 3.11, you can download it from the [official Python website](https://www.python.org) or use a package manager.

Optionally, you can create a virtual environment. Then install all the necessary dependencies. An example with [Mamba](https://mamba.readthedocs.io/en/latest/index.html) is:
```bash
# Optionally create a new Mamba environment with Python 3.11 and specify a name
mamba create -n <YOUR_ENV_NAME> python=3.11

# Activate the Mamba environment
mamba activate <YOUR_ENV_NAME>
```

Follow NVIDIA Conda CUDA installation steps below:
```bash
# Make sure GPU available
lspci | grep -i nvidia

mamba install cuda -c nvidia
```

Once your environment has been prepared, install all required packages:
```bash
pip install -r requirements.txt
```

### 2: YAML Configuration Files

This project uses YAML configuration files to store all pipeline parameters and paths. The design choice of the YAML file is intended to eliminate repetition of commonly used parameters across code, as well as simplify future changes and refactors, allow developers to add new parameters, and make all settings visible to the user in one consolidated place.

To prepare a YAML config file, copy [template_config.yaml](./configs/template_config.yaml) into the [user_configs](./configs/user_configs/) folder. Fill out all parameters accordingly. Absolute paths are preferred for any path variables, but the repository is set up to work flexibly with any desired directory structure.

> [!WARNING]  
> In most cases, YAML does not expect strings. Adding quotation marks around arguments in the config file can lead to unexpected errors.

### 3: Script Execution
> [!TIP]  
> Once a YAML config file is prepared, it can be passed into any script in the pipeline.

The Python scripts for data preprocessing and training can be run via the terminal.

The expected order of script execution is as follows:
1. Change current directory to [src](./src/).
2. Run [vanilla_rag.py](./src/vanilla_rag.py).

For example, if you want run basic RAG inference:

```bash
cd src/

python vanilla_rag.py /home/<path_to_repo>/RISC-2024-CO/configs/user_configs/<your_config_name>.yaml
```
