# donkeycar: a python self driving library

[![Build Status](https://travis-ci.org/autorope/donkeycar.svg?branch=dev)](https://travis-ci.org/autorope/donkeycar)
[![CodeCov](https://codecov.io/gh/autoropoe/donkeycar/branch/dev/graph/badge.svg)](https://codecov.io/gh/autorope/donkeycar/branch/dev)
[![PyPI version](https://badge.fury.io/py/donkeycar.svg)](https://badge.fury.io/py/donkeycar)
[![Py versions](https://img.shields.io/pypi/pyversions/donkeycar.svg)](https://img.shields.io/pypi/pyversions/donkeycar.svg)

Donkeycar is minimalist and modular self driving library for Python. It is
developed for hobbyists and students with a focus on allowing fast experimentation and easy
community contributions.

#### Quick Links
* [Donkeycar Updates & Examples](http://donkeycar.com)
* [Build instructions and Software documentation](http://docs.donkeycar.com)
* [Slack / Chat](https://donkey-slackin.herokuapp.com/)

![donkeycar](./docs/assets/build_hardware/donkey2.png)

#### Use Donkey if you want to:
* Make an RC car drive its self.
* Compete in self driving races like [DIY Robocars](http://diyrobocars.com)
* Experiment with autopilots, mapping computer vision and neural networks.
* Log sensor data. (images, user inputs, sensor readings)
* Drive your car via a web or game controller.
* Leverage community contributed driving data.
* Use existing CAD models for design upgrades.

### Get driving.
After building a Donkey2 you can turn on your car and go to http://localhost:8887 to drive.

### Modify your cars behavior.
The donkey car is controlled by running a sequence of events

```python
#Define a vehicle to take and record pictures 10 times per second.

import time
from donkeycar import Vehicle
from donkeycar.parts.cv import CvCam
from donkeycar.parts.datastore import TubWriter
V = Vehicle()

IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3

#Add a camera part
cam = CvCam(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH)
V.add(cam, outputs=['image'], threaded=True)

#warmup camera
while cam.run() is None:
    time.sleep(1)

#add tub part to record images
tub = TubWriter(path='./dat',
          inputs=['image'],
          types=['image_array'])
V.add(tub, inputs=['image'], outputs=['num_records'])

#start the drive loop at 10 Hz
V.start(rate_hz=10)
```

See [home page](http://donkeycar.com), [docs](http://docs.donkeycar.com)
or join the [Discord server](http://www.donkeycar.com/community.html) to learn more.

# Setting up Donkeycar and Donkey Simulator for Reinforcement Learning

### Requirements:
* Donkeycar with a RaspberryPi 4 board
* IntelRealsense T265 tracking camera and the Python wrapper for the driver library installed on the RaspberryPi 4.
* An additional computational resource with a GPU for training the RL models.

### Installation:

* Install this branch of donkeycar using the standard installation [instructions](http://docs.donkeycar.com/guide/install_software/) on the Donkeycar and on a GPU machine.
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
