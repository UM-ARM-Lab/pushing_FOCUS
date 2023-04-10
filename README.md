# Pushing FOCUS

An example of the FOCUS method, applied to a planar pushing task.

# Motivation

When I was planning my presentation at ICRA 2023, I realized both of the examples I show of FOCUS involve avoiding contact.
This is definitely a strong use-case for FOCUS, but not a particularly exciting one.
Manipulation is **all about contact**, so it would be a shame if FOCUS could only be applied to avoiding it.

Therefore, this example aims to show how FOCUS can be used to adapt more quickly from pushing a cube to pushing a cylinder.
This means we *are* modeling and adapting contact dynamics, but still avoiding adapting to new dynamics and improving data efficiency.

The figure below explains the adaptation problem:

TODO: figure

More details about FOCUS can be found on our [Project Website](https://sites.google.com/view/focused-adaptation-dynamics/home)!

# How to run

## Setup
1. Set up a virtual environment with python 3.10 and install the dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Steps to Reproduce

```bash
# Collect the source dataset.
./collect_data.py source.xml

# Collect the target dataset.
./collect_data.py target.xml

# Train the source dynamics model.
./train_source.py

# Run a baseline adaptation method
./adapt.py all_data

# Run FOCUS
./adapt.py FOCUS

# Compare the methods in [Weights & Biases](https://wandb.ai/).
```

## Results

TODO: Link to wandb report here!