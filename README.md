# Pyquaticus
This is a [PettingZoo](https://pettingzoo.farama.org/) environment for maritime Capture the Flag with uncrewed surface vehicles (USVs).

## Motivation
This PettingZoo is a _lightweight_ environment for developing algorithms to play multi-agent Capture-the-Flag with surface vehicle dynamics.

The primary motivation is to enable quick iteration of algorithm development or training loops for Reinforcement Learning (RL). The implementation is pure-Python and supports faster-than-real-time execution and can easily be parallelized on a cluster. This is critical for scaling up learning-based techniques before transitioning to higher-fidelity simulation and/or USV hardware.

The default vehicle dynamics are based on the [MOOS-IvP](https://oceanai.mit.edu/moos-ivp/pmwiki/pmwiki.php?n=Main.HomePage) `uSimMarine` dynamics [here](https://oceanai.mit.edu/ivpman/pmwiki/pmwiki.php?n=IvPTools.USimMarine). MOOS-IvP stands for Mission Oriented Operating Suite with Interval Programming. The IvP addition to the core MOOS software package is developed and maintained by the Laboratory for Autonomous Marine Sensing Systems at MIT. MOOS-IvP is a popular choice for maritime autonomy filling a similar role to the Robot Operating System (ROS) used in other robotics fields.

## Key Capabilities
* Supports standard PettingZoo interface for multi-agent RL
* Pure-Python implementation without many dependencies
* Easy integration with standard learning frameworks
* Implementation of MOOS-IvP vehicle dynamics such that algorithms and learning policies can easily be ported to MOOS-IvP and deployed on hardware
* Baseline policies to train against
* Parameterized number of agents
* Configurable observation space
* Decentralized and agent-relative observation space
* Configurable reward function
* Supports custom agent dynamics
* Simulate real-world maritime scenarios of any aquatic region on earth with [OpenStreetMap](https://www.openstreetmap.org/)-based environments
* Example integration with [RLLib](https://docs.ray.io/en/latest/rllib/index.html) for reinforcement learning  


## Installation
It is highly recommended to use a `conda` environment. Assuming you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed, run the following from the top-level of this repository:

```
# create a small virtual environment -- just enough to run the PettingZoo environment
./setup-conda-env.sh light
```

```
# or create the full virtual environment -- with RLLib and PyTorch
./setup-conda-env.sh full
```

You can then activate the environment with: `conda activate env-light/` or `conda activate env-full/` 

## Basic Tests

* Random action: `python ./test/rand_env_test.py`
* Control with arrow keys: `python ./test/arrowkeys_test.py`
  * control agents with WASD and the arrow keys

## Environment Visuals

* Rendered with `pygame`
* Blue vs Red 1v1 or multiagent teams
* **Flag keepout zone:** circle (team's color) drawn around flag
* **Flag pickup zone:** black circle drawn around flag
* **Tagging cooldown**: receding black circle around agent
* **Out-of-bounds**: yellow halo around agent (occurs if out-of-bounds)
* **Drive-to-home**: green halo around agent (occurs if tagged)
* **Lines between agents:**
  * Drawn between agents of opposite teams
  * **Green**: within `2*catch_radius`
  * **Orange/Yellow**: within `1.5*catch_radius`
  * **Red**: within `catch_radius`

## Configurable Reward

Pyquaticus comes with a simple sparse reward, but it can be extended with different reward structures. See [rewards.py](https://github.com/mit-ll-trusted-autonomy/pyquaticus/blob/main/pyquaticus/utils/rewards.py) for more information.

## Docker

Out-of-date, do not use. Coming soon!

## Distribution and Disclaimer Statements

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

