import sys
import numpy as np
from donkeycar.parts.network import MQTTValuePub, MQTTValueSub

import time

sys.path.insert(1, "/home/ari/Documents/RLDonkeyCar")
sys.path.insert(1, "/u/70/viitala1/unix/Documents/Dippa/RLDonkeyCar")
sys.path.insert(1, "/home/pi/Documents/RLDonkeyCar")


from models.ae_sac import AE_SAC
#from gym import spaces

DONKEY_NAME = "Kari_real"

STEER_LIMIT_LEFT = -1
STEER_LIMIT_RIGHT = 1
THROTTLE_MAX = 0.6
THROTTLE_MIN = 0.25
MAX_STEERING_DIFF = 0.25
STEP_LENGTH = 0.1
RANDOM_EPISODES = 1
GRADIENT_STEPS = 600

MAX_EPISODE_STEPS = 1000

COMMAND_HISTORY_LENGTH = 5
FRAME_STACK = 1
VAE_OUTPUT = 20
LR = 0.0001

IMAGE_SIZE = 40
RGB = False

SKIP_INITIAL_STEPS = 20
# action_space = spaces.Box(
#     low=np.array([STEER_LIMIT_LEFT, THROTTLE_MIN]), 
#     high=np.array([STEER_LIMIT_RIGHT, THROTTLE_MAX]), dtype=np.float32)


PARAMS = {

    "sac": {
        "linear_output": VAE_OUTPUT + COMMAND_HISTORY_LENGTH * 3,
        "lr": LR,
        "target_entropy": -2,
        "batch_size": 128,
        "hidden_size": 64,
        "encoder_update_frequency": 0,
        "critic_loss_encoder_update": True,
        "pretrained_ae": "",
        "im_size": IMAGE_SIZE,
        "n_images": 20000,
        "epochs": 1000
        },
    "ae": {
        "framestack": FRAME_STACK,
        "output": VAE_OUTPUT,
        "linear_input": 100,
        "image_size": IMAGE_SIZE,
        "lr": LR / 10,
        "image_channels": 3 if RGB else 1,
        "encoder_type": "vae",
        "batch_size": 64,
        "l2_regularization": False
    }
}

class RL_Agent():
    def __init__(self, alg_type, sim):
        self.agent = AE_SAC(PARAMS)
        self.sim = sim

        self.image = np.zeros((120, 160, 3))
        self.command_history = np.zeros(3*COMMAND_HISTORY_LENGTH)

        self.state = np.vstack([self.agent.process_im(self.image, IMAGE_SIZE, RGB) for x in range(FRAME_STACK)])
        self.speed = 0

        self.step = 0
        self.episode = 0
        self.episode_reward = 0
        self.replay_buffer = []

        self.target_speed = 0
        self.steering = 0

        self.training = False
        self.step_start = 0

        self.replay_buffer_pub = MQTTValuePub(DONKEY_NAME + "buffer", broker="mqtt.eclipse.org")
        self.replay_buffer_sub = MQTTValueSub(DONKEY_NAME + "buffer", broker="mqtt.eclipse.org")

        self.param_pub = MQTTValuePub(DONKEY_NAME + "param", broker="mqtt.eclipse.org")
        self.param_sub = MQTTValueSub(DONKEY_NAME + "param", broker="mqtt.eclipse.org")
        
        self.params = self.param_sub.run()

    def reset(self, image):
        self.episode += 1

        self.episode_reward = 0
        self.replay_buffer = []

        self.target_speed = 0
        self.steering = 0

        self.command_history = np.zeros(3*COMMAND_HISTORY_LENGTH)
        self.state = np.vstack([image for x in range(FRAME_STACK)])
        self.buffer_sent = False
        self.buffer_received = False
        self.params_sent = False
        self.params_received = False


    def train(self):
        #print(f"Training for {int(time.time() - self.training_start)} seconds")    

        if not self.buffer_sent or (time.time() - self.training_start) > 60:
            print("Buffer sent")
            self.replay_buffer_pub.run(self.replay_buffer[SKIP_INITIAL_STEPS:])
            self.buffer_sent = True

        new_params = self.param_sub.run()
        
        if not new_params:
            return True

        print("Received new params.")
        self.replay_buffer_pub.run(False)
        self.agent.import_parameters(new_params)
        
        self.buffer_sent = False

        return False


    def run(self, image, speed=None):

        if not speed:
            self.speed = self.target_speed
        else:
            self.speed = speed

        if image is not None:
            self.image = image

        self.dead = self.is_dead(self.image) if not self.sim else self.is_dead_sim(self.image)

        if self.step > 0 and not self.training:
            """Save observation to replay buffer"""
            reward = 1 + (self.speed - THROTTLE_MIN) / (THROTTLE_MAX - THROTTLE_MIN)
            done = self.dead
            reward = reward * -10 if self.dead else reward

            next_command_history = np.roll(self.command_history, 3)
            next_command_history[:3] = [self.steering, self.target_speed, self.speed]

            next_state = np.roll(self.state, 1)
            next_state[:1, :, :] = self.agent.process_im(self.image, IMAGE_SIZE, RGB)

            self.replay_buffer.append([ [self.state, self.command_history], 
                                        [self.steering, self.target_speed],
                                        [reward],
                                        [next_state, next_command_history],
                                        [float(not done)]])

            self.episode_reward += reward
            step_end = time.time()

            self.state = next_state
            self.command_history = next_command_history

            print(f"Episode: {self.episode}, Step: {self.step}, Reward: {reward:.2f}, Episode reward: {self.episode_reward:.2f}, Step time: {(self.step_start - step_end):.2f}, Speed: {self.speed:.2f}")


        if self.step > MAX_EPISODE_STEPS or (self.dead and not self.training):
            self.training_start = time.time()
        
            self.step = 0
            self.steering = 0
            self.target_speed = 0
            self.training = True

            return self.steering, self.target_speed, self.training


        if self.training:
        
            self.training = self.train()
            self.dead = False

            
            return self.steering, self.target_speed, self.training


        if self.step == 0:
            if not self.sim:
                input("Press Enter to start a new episode.")
            
            self.reset(self.agent.process_im(self.image, IMAGE_SIZE, RGB))

        self.step += 1
        
        if self.step < SKIP_INITIAL_STEPS:
            return 0, 0, False

        self.step_start = time.time()

        #if self.episode < RANDOM_EPISODES:
        #    action = action_space.sample()
        #else:
            
        action = self.agent.select_action((self.state, self.command_history))

        self.steering, self.target_speed = self.enforce_limits(action, self.command_history[0]) 

        return self.steering, self.target_speed, self.training


    def is_dead(self, img):
        """
        Counts the black pixels from the ground and compares the amount to a threshold value.
        If there are not enough black pixels the car is assumed to be off the track.
        """

        crop_height = 20
        crop_width = 20
        threshold = 70
        pixels_percentage = 0.10

        pixels_required = (img.shape[1] - 2 * crop_width) * crop_height * pixels_percentage

        crop = img[-crop_height:, crop_width:-crop_width]

        r = crop[:,:,0] < threshold
        g = crop[:,:,1] < threshold
        b = crop[:,:,2] < threshold

        pixels = (r & g & b).sum()

        #print("Pixels: {}, Required: {}".format(pixels, pixels_required))
        
        return  pixels < pixels_required

    def is_dead_sim(self, img):

        crop_height = 40
        required = 0.8
        
        cropped = img[-crop_height:]

        rgb = cropped[:,:,0] > cropped[:,:,2]

        return rgb.sum() / (crop_height * 160) > required

    def enforce_limits(self, action, prev_steering):
        """
        Scale the agent actions to environment limits
        """

        var = (THROTTLE_MAX - THROTTLE_MIN) / 2
        mu = (THROTTLE_MAX + THROTTLE_MIN) / 2

        steering_min = max(STEER_LIMIT_LEFT, prev_steering - MAX_STEERING_DIFF)
        steering_max = min(STEER_LIMIT_RIGHT, prev_steering + MAX_STEERING_DIFF)

        steering = max(steering_min, min(steering_max, action[0]))

        return [steering, action[1] * var + mu]

if __name__ == "__main__":
    print("Starting as training server")
    agent = RL_Agent("sac", False)
    params_sent = False
    buffer_received = False
    trained = False

    while True:
        new_buffer = agent.replay_buffer_sub.run()

        if new_buffer and not trained:
            print(f"{len(new_buffer)} new buffer observations")
            agent.agent.append_buffer(new_buffer)
            print("Training")
            agent.agent.update_parameters(GRADIENT_STEPS)
            params = agent.agent.export_parameters()
            trained = True
        
        if trained:
            print("Sending parameters")
            agent.param_pub.run(params)
        
        if not new_buffer:
            agent.param_pub.run(False)
            trained = False
            print("Waiting for observations.")

        time.sleep(1)






