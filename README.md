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
./src/collect_source.py
./src/collect_target.py
./src/collect_similar.py
./src/collect_dissimilar.py

# Train the source dynamics model.
./src/train_source.py

# visualize predictions
./src/viz_predictions.py source-dataset # or target-dataset, similar-dataset, dissimilar-dataset

# Run adaptation methods
./src/adapt.py all_data
./src/adapt.py FOCUS

# [OPTIONAL] visualize the adaptation process on the target dataset,
# and the similar and dissimilar datasets for FOCUS and two baselines.
./src/animate_adapt.py

# Run a simple LQR controller to reach a position target in the source environment
# using the source dynamics
./src/source_control.py
```

## Results


## Animations

See the `animations/` folder for videos of the adaptation process for FOCUS versus two baselines.