'''
file: manage_remote.py
author: Tawn Kramer
date: 2019-01-24
desc: Control a remote donkey robot over network
'''
import time
import math
import donkeycar as dk
from donkeycar.parts.camera import PiCamera
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.parts.network import MQTTValueSub, MQTTValuePub
from donkeycar.parts.image import ImgArrToJpg
from donkeycar.parts.rl_agent import RL_Agent
from donkeycar.parts.episode_logger import EpisodeLogger

cfg = dk.load_config()

V = dk.Vehicle()

print("starting up", cfg.DONKEY_UNIQUE_NAME, "for remote management.")

class Constant:
    def run(self, image):
        print(image)
        return 1, 1

#CAMERA

if cfg.DONKEY_GYM:
    from donkeycar.parts.dgym import DonkeyGymEnv 
    cam = DonkeyGymEnv(cfg.DONKEY_SIM_PATH, host=cfg.SIM_HOST, env_name=cfg.DONKEY_GYM_ENV_NAME, conf=cfg.GYM_CONF, delay=cfg.SIM_ARTIFICIAL_LATENCY)
    inputs = ["steering", 'target_speed']

    logger = EpisodeLogger()
    V.add(logger, inputs=["training", "steering", "target_speed", "pos", "vel"])

else:
    inputs = []
    cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)

V.add(cam, inputs=inputs, outputs=["image"], threaded=True)

#V.add(Constant(), inputs=["image"], outputs=["steering", "target_speed"])

agent = RL_Agent(alg_type=cfg.RL_ALG_TYPE, sim=cfg.DONKEY_GYM)
V.add(agent, inputs=["image", "speed"], outputs=["steering", "target_speed", "training"], threaded=False)



#REALSENSE

if cfg.REALSENSE:

    from donkeycar.parts.realsense2 import RS_T265
    from donkeycar.parts.pid import PID

    rs = RS_T265()
    V.add(rs, outputs=["pos", "vel", "acc", "img", "speed"], inputs= ["training"], threaded=True)

    pid = PID()
    V.add(pid, inputs=["target_speed", "speed", "training"], outputs=["throttle"])


#STEERING 

if not cfg.DONKEY_GYM:

    steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
    steering = PWMSteering(controller=steering_controller,
                                    left_pulse=cfg.STEERING_LEFT_PWM, 
                                    right_pulse=cfg.STEERING_RIGHT_PWM)

    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
    throttle = PWMThrottle(controller=throttle_controller,
                                    max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                    zero_pulse=cfg.THROTTLE_STOPPED_PWM, 
                                    min_pulse=cfg.THROTTLE_REVERSE_PWM)

    V.add(steering, inputs=['steering'])
    V.add(throttle, inputs=['throttle'])


V.start(rate_hz=cfg.DRIVE_LOOP_HZ)
