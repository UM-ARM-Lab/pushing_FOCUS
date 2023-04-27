# Pushing FOCUS

An example of the FOCUS method, applied to a planar pushing task.

# Motivation

When I was planning my presentation at ICRA 2023, I realized both of the examples I show of FOCUS involve avoiding contact.
This is definitely a strong use-case for FOCUS.
Sometimes, avoiding modeling certain contact dynamics makes a lot of sense, however I wanted an example where at least _some_ contact dynamics _were_ learned.

Therefore, this example aims to show how FOCUS can be used to adapt in a simple planar pushing task.
This means we *are* modeling and adapting contact dynamics, but still avoiding adapting to new dynamics and improving data efficiency.

The figure below explains the adaptation problem:

TODO: figure

More details about FOCUS can be found on our [Project Website](https://sites.google.com/view/focused-adaptation-dynamics/home)!

# How to run

## Setup

1. Clone this repository.

1. Set up a virtual environment with python 3.10 and install the dependencies. On Ubuntu 20.04 you will have to install python3.10-venv from the `deadsnakes` PPA.

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Visualization

To see the simulation running, launch Rerun.

```rerun
rerun --memory-limit 8GB
```

The memory limit will cause Rerun to drop old messages once that memory limit is reached.

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

# See how MPPI works in the source environment with the learned dynamics
./src/source_control.py

# Now run FOCUS in the target environment
./src/full_online_adaptation.py --method=FOCUS
```

## Results


## Animations

See the `animations/` folder for videos of the adaptation process for FOCUS versus two baselines.
