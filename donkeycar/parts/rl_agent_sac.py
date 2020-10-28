import sys
import time
import argparse
import copy
import os

import numpy as np
import torch

from donkeycar.parts.network import MQTTValuePub, MQTTValueSub

sys.path.insert(1, "/home/ari/Documents/RLDonkeyCar")
sys.path.insert(1, "/u/70/viitala1/unix/Documents/Dippa/RLDonkeyCar")
sys.path.insert(1, "/home/pi/Documents/RLDonkeyCar")

from models.ae_sac import AE_SAC

parser = argparse.ArgumentParser()

parser.add_argument("--car_name", help="Name of the car on MQTT-server", default="Kari")
parser.add_argument("--episode_steps", help="Number of steps per episode", default=1000, type=int)
parser.add_argument("--episodes", help="Number of steps episodes per run", default=100, type=int)
parser.add_argument("--encoder_update", help="Type of encoder to be used", default="aesac")
parser.add_argument("--total_steps", help="Max steps for a run", default=50000, type=int)
parser.add_argument("--runs", help="How many runs to do", default=10, type=int)
parser.add_argument("--load_model", help="Load pretrained model", default="")
parser.add_argument("--save_model", help="File name to save model", default="")
parser.add_argument("--update_loaded_model", action="store_true", help="Update the model")

args = parser.parse_args()

if args.save_model and not os.path.isdir("./models"):
    os.mkdir("./models")

MODEL_PATH = f"./models/{args.save_model}.pth"

DONKEY_NAME = args.car_name

STEER_LIMIT_LEFT = -1
STEER_LIMIT_RIGHT = 1
THROTTLE_MAX = 1.2
THROTTLE_MIN = 0.2
MAX_STEERING_DIFF = 2
MAX_THROTTLE_DIFF = 0.3
STEP_LENGTH = 0.1
RANDOM_EPISODES = 1
GRADIENT_STEPS = 600

SKIP_INITIAL_STEPS = 20
BLOCK_SIZE = 200
MAX_EPISODE_STEPS = args.episode_steps + SKIP_INITIAL_STEPS
TRAINING_TIMEOUT = 200

COMMAND_HISTORY_LENGTH = 5
FRAME_STACK = 1
VAE_OUTPUT = 20
LR = 0.0001

IMAGE_SIZE = 40
RGB = False
IMAGE_CHANNELS = 3 if RGB else 1

PARAMS = {

    "sac": {
        "linear_output": VAE_OUTPUT + COMMAND_HISTORY_LENGTH * 3,
        "lr": LR,
        "target_entropy": -2,
        "batch_size": 128,
        "hidden_size": 64,
        "encoder_update_frequency": 1,
        "encoder_critic_loss": True,
        "encoder_ae_loss": True,
        "pretrained_ae": "",
        "im_size": IMAGE_SIZE,
        "n_images": 20000,
        "epochs": 1000,
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
    def __init__(self, alg_type, sim, car_name=args.car_name, encoder_update=args.encoder_update, max_episode_steps=MAX_EPISODE_STEPS):

        if encoder_update == "pixel":
            print("Using Pixel encoder")
            PARAMS["sac"]["encoder_critic_loss"] = True
            PARAMS["sac"]["encoder_ae_loss"] = False

        if encoder_update == "ae":
            print("Using Autoencoder loss")
            PARAMS["sac"]["encoder_critic_loss"] = False
            PARAMS["sac"]["encoder_ae_loss"] = True

        if encoder_update == "aesac":
            print("Using autoencoder and critic loss")
            PARAMS["sac"]["encoder_critic_loss"] = True
            PARAMS["sac"]["encoder_ae_loss"] = True

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
        self.max_episode_steps = max_episode_steps

        self.target_speed = 0
        self.steering = 0

        self.training = False
        self.step_start = 0

        self.buffers_sent = False

        self.replay_buffer_pub = MQTTValuePub(car_name + "buffer", broker="mqtt.eclipse.org")
        self.replay_buffer_sub = MQTTValueSub(car_name + "buffer", broker="mqtt.eclipse.org", def_value=(0, True))

        self.replay_buffer_received_pub = MQTTValuePub(car_name + "buffer_received", broker="mqtt.eclipse.org")
        self.replay_buffer_received_sub = MQTTValueSub(car_name + "buffer_received", broker="mqtt.eclipse.org", def_value=0)

        self.param_pub = MQTTValuePub(car_name + "param", broker="mqtt.eclipse.org")
        self.param_sub = MQTTValueSub(car_name + "param", broker="mqtt.eclipse.org")


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

        if (time.time() - self.training_start) > TRAINING_TIMEOUT:
            """Temporary fix for when sometimes the replay buffer fails to send"""
            self.training_start = time.time()
            self.buffers_sent = 0
            self.replay_buffer_pub.run((0, False))
            return False

        if len(self.replay_buffer) > 0:

            buffers_received = self.replay_buffer_received_sub.run()

            if self.buffers_sent == buffers_received:
                self.buffers_sent += 1
                self.replay_buffer_pub.run((self.buffers_sent, self.replay_buffer[:BLOCK_SIZE]))
                print(f"Sent {len(self.replay_buffer[:BLOCK_SIZE])} observations")
                self.replay_buffer = self.replay_buffer[BLOCK_SIZE:]
                
            return True

        if self.replay_buffer_received_sub.run() == self.buffers_sent:
            self.buffers_sent = 0
            self.replay_buffer_received_pub.run(0)
            self.replay_buffer_pub.run((0, False))


        new_params = self.param_sub.run()
        
        if not new_params:
            return True

        print("Received new params.")
        self.agent.import_parameters(new_params)
        self.param_pub.run(False)

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
            if THROTTLE_MAX == THROTTLE_MIN:
                reward = 1
            else:
                reward = 1 + (self.speed - THROTTLE_MIN) / (THROTTLE_MAX - THROTTLE_MIN)
                reward = min(reward, 2) / 2

            done = self.dead
            reward = reward * -10 if self.dead else reward

            next_command_history = np.roll(self.command_history, 3)
            next_command_history[:3] = [self.steering, self.target_speed, self.speed]

            next_state = np.roll(self.state, IMAGE_CHANNELS)
            next_state[:IMAGE_CHANNELS, :, :] = self.agent.process_im(self.image, IMAGE_SIZE, RGB)

            self.replay_buffer.append([ [self.state, self.command_history], 
                                        [self.steering, self.target_speed],
                                        [reward],
                                        [next_state, next_command_history],
                                        [float(not done)]])

            self.episode_reward += reward
            step_end = time.time()

            self.state = next_state
            self.command_history = next_command_history

            print(f"Episode: {self.episode}, Step: {self.step}, Reward: {reward:.2f}, Episode reward: {self.episode_reward:.2f}, Step time: {(self.step_start - step_end):.2f}, Speed: {self.speed:.2f}, Throttle: {self.target_speed:.2f}")


        if self.step > self.max_episode_steps or (self.dead and not self.training):
            self.training_start = time.time()
        
            self.step = 0
            self.steering = 0
            self.target_speed = 0
            self.training = True
            self.replay_buffer = self.replay_buffer[SKIP_INITIAL_STEPS:]
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
            return 0.001, 0, False

        self.step_start = time.time()

        #if self.episode < RANDOM_EPISODES:
        #    action = action_space.sample()
        #else:
            
        action = self.agent.select_action((self.state, self.command_history))

        self.steering, self.target_speed = self.enforce_limits(action, self.command_history[:2]) 

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

    def enforce_limits(self, action, prev_action):
        """
        Scale the agent actions to environment limits
        """

        prev_steering = prev_action[0]
        prev_throttle = prev_action[1]

        var = (THROTTLE_MAX - THROTTLE_MIN) / 2
        mu = (THROTTLE_MAX + THROTTLE_MIN) / 2

        raw_throttle = action[1] * var + mu

        steering = np.clip(action[0], prev_steering - MAX_STEERING_DIFF, prev_steering + MAX_STEERING_DIFF)
        steering = np.clip(steering, STEER_LIMIT_LEFT, STEER_LIMIT_RIGHT)

        throttle = np.clip(raw_throttle, prev_throttle - MAX_THROTTLE_DIFF, prev_throttle + MAX_THROTTLE_DIFF)
        
        return [steering, throttle]

if __name__ == "__main__":
    print("Starting as training server")
    agent = RL_Agent("sac", False, DONKEY_NAME, encoder_update=args.encoder_update)
    
    if args.load_model:
        agent.agent = torch.load(args.load_model)

    params_sent = False
    buffer_received = False
    trained = False
    training_episodes = 0
    buffers_received = 0
    prev_buffer = 0
    runs = 1

    while training_episodes < args.episodes:
        new_buffer = agent.replay_buffer_sub.run()
        
        if (new_buffer[0] - 1)  == prev_buffer and not trained:
            print("New buffer")
            print(f"{len(new_buffer[1])} new buffer observations")
            agent.agent.append_buffer(new_buffer[1])
            prev_buffer += 1
            agent.replay_buffer_received_pub.run(prev_buffer)

        if new_buffer[1] == False and prev_buffer > 0 and not trained:
            
            if not args.load_model or args.update_loaded_model:
                print("Training")
                agent.agent.update_parameters(GRADIENT_STEPS)
                if len(agent.agent.replay_buffer.buffer) > args.total_steps:
                    agent = RL_Agent("sac", False, DONKEY_NAME, encoder_update=args.encoder_update)
                    runs += 1
                    if runs > args.runs:
                        break
            
            if args.save_model:
                print("Saving model")
                agent_2 = copy.deepcopy(agent.agent)
                agent_2.replay_buffer.buffer = []
                torch.save(agent_2, MODEL_PATH)

            params = agent.agent.export_parameters()
            trained = True
            print("Sending parameters")
            agent.param_pub.run(params)
            time.sleep(1)
        
        if trained and agent.param_sub.run() == False:
            trained = False
            prev_buffer = 0
            print("Waiting for observations.")
            print(f"Run {runs}")

        time.sleep(0.1)






