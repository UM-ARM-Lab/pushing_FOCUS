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
# Collect the datasets
./collect_source.py
./collect_target.py
./collect_similar.py
./collect_dissimilar.py

# Train the source dynamics model.
./train_source.py

# visualize predictions
./viz_predictions.py source-dataset # or target-dataset, similar-dataset, dissimilar-dataset

# Run adaptation methods
./adapt.py all_data
./adapt.py FOCUS

# evaluate on the similar and dissimilar datasets
./evaluate.py adapted_FOCUS_model
./evaluate.py adapted_all_data_model

# Compare the methods in [Weights & Biases](https://wandb.ai/).
```

## Results

TODO: Link to wandb report here!