# Setting up Donkeycar and Donkey Simulator for Reinforcement Learning

### Requirements:
* Donkeycar with a RaspberryPi 4 board
* IntelRealsense T265 tracking camera and the Python wrapper for the driver library installed on the RaspberryPi 4.
* An additional computational resource with a GPU for training the RL models.

### Installation:

* Install this fork of donkeycar-package using the standard installation [instructions](http://docs.donkeycar.com/guide/install_software/) on the Donkeycar and on a GPU machine.
* If you want to use the simulator, additionally install the [simulator pacakage](http://docs.donkeycar.com/guide/simulator/).
* Clone repos containing the [SAC](https://github.com/ari-viitala/RLDonkeyCar) and [Dreamer](https://github.com/AaltoVision/donkeycar-dreamer) agents.
* Edit `donkeycar/donkeycar/parts/rl_agent_sac.py` and `donkeycar/donkeycar/parts/rl_agent_dreamer.py` files such that they can import SAC and Dreamer modules respectively.
* On the Donkeycar create a donkeycar application by `donkey createcar --path ~/donkey_rl --template rl_local_control`.
* If you are using the simulator, edit `~/donkey_rl/myconfig.py` according to the simulator instructions.

### Using SAC

Start the training server on the GPU machine
```
cd donkeycar/donkeycar/parts
conda activate donkey
python rl_agent_sac.py
```
On the Donkeycar run
```
cd donkey_rl
python manage.py
```
### Using Dreamer

Start the training server on the GPU machine
```
cd donkeycar/donkeycar/parts
conda activate donkey
python rl_agent_dreamer.py
```
On the Donkeycar

```
cd donkey_rl
```

Edit the `myconfig.py` file and change `RL_ALG_TYPE = "Dreamer"` and then run

```
python manage.py
```

The car should start driving. When the episode ends, the collected observations are sent over to the GPU machine and after training completes the updated weights are sent back to the car and a new episode can be started.

### Notes

In this application the episode termination is done using a computer vision approach that detects wheter the car is on the track or not and terminates the episode accordingly. To use this feature you need a dark/black track on a sufficiently light colored surface. The materials used in our experiments were black plastic and a birch wood floor and `donkey-generated-roads-v0` environment on the simulator.
