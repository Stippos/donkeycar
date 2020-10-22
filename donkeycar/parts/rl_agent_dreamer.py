import sys
import time
import argparse
import copy
import os

import numpy as np
import torch

from donkeycar.parts.network import MQTTValuePub, MQTTValueSub

sys.path.insert(1, "/u/95/zhaoy13/unix/ICRA/donkeycar-dreamer")
sys.path.insert(1, "/home/ari/Documents/donkeycar-dreamer")
sys.path.insert(1, "/u/70/viitala1/unix/Documents/Dippa/donkeycar-dreamer")
sys.path.insert(1, "/home/pi/Documents/donkeycar-dreamer")

from agent import Dreamer
import torch
#import wandb

parser = argparse.ArgumentParser()

parser.add_argument("--car_name", help="Name of the car on MQTT-server", default="Kari")
parser.add_argument("--episode_steps", help="Number of steps per episode", default=1000, type=int)
parser.add_argument("--episodes", help="Number of steps episodes per run", default=100, type=int)
parser.add_argument("--encoder_update", help="Type of encoder to be used", default="aesac")
parser.add_argument("--total_steps", help="Max steps for a run", default=50000, type=int)
parser.add_argument("--runs", help="How many runs to do", default=10, type=int)
parser.add_argument("--load_model", help="Load pretrained model", default="")
parser.add_argument("--save_model", help="File name to save model", default="")


args = parser.parse_args()

if args.save_model and not os.path.isdir("./models"):
    os.mkdir("./models")

MODEL_PATH = f"./models/{args.save_model}.pth"
LOAD_MODEL = args.load_model
SAVE_MODEL = args.save_model

#DONKEY_NAME = args.car_name
TRAINING_TIMEOUT = 300
BLOCK_SIZE = 200

class AttrDict(dict):
	__setattr__ = dict.__setitem__
	__getattr__ = dict.__getitem__


def define_config():
	args = AttrDict()
	# parameter for dreamer
	args.car_name = "ari_dreamer"
	args.episodes_steps = 1000
	args.episodes = 100

	args.belief_size = 200
	args.state_size = 30
	args.hidden_size = 300
	args.embedding_size = 1024
	args.observation_size = (1, 40, 40)  # TODO: change this latter
	args.action_size = 2  # TODO: change this latter
	args.device = "cuda" if torch.cuda.is_available() else "cpu"
	args.testing_device = "cpu"
	args.symbolic = False
	args.dense_act = 'elu'
	args.cnn_act = 'relu'

	args.pcont_scale = 5
	args.reward_scale = 5
	args.world_lr = 6e-4
	args.actor_lr = 8e-5
	args.value_lr = 8e-5
	args.free_nats = 3
	args.experience_size = 1000000
	args.bit_depth = 5
	args.discount = 0.99
	args.temp = 0.2  # entropy temperature
	args.disclam = 0.95
	args.planning_horizon = 15
	args.batch_size = 50
	args.chunk_size = 50
	args.grad_clip_norm = 100.0

	args.expl_amount = 0.3  # action noise

	# for action constrains
	args.throttle_base = 0.3  # fixed throttle base
	args.throttle_min = 0.1
	args.throttle_max = 0.5
	args.angle_min = -1
	args.angle_max = 1
	# I didn't limit the max steering_diff yet
	args.max_steering_diff = 0.25
	args.step_length = 0.1

	# add prefill episodes
	args.prefill_episodes = 5
	args.random_episodes = 6
	args.gradient_steps = 100
	args.skip_initial_steps = 20
	args.block_size = 200

	args.max_episodes_steps = args.episodes_steps + args.skip_initial_steps

	# set up for experiments
	args.pcont = False  # whether to use a learned pcont
	args.with_logprob = False  # whether to use the soft actor-critic
	args.fix_speed = True  # whether to use fixed speed, fixed speed is 0.3

	args.temp = 0.003
	return args

class RL_Agent():
    def __init__(self, alg_type, sim, car_name=args.car_name):
        self.args = define_config()
        self.agent = Dreamer(self.args)
        self.sim = sim

        self.image = np.zeros((120, 160, 3))
        self.observation = torch.zeros((1, 3, 64, 64))  # init observation, with batch dim
        self.belief = torch.zeros(1, self.args.belief_size, device=self.args.device)
        self.posterior_state = torch.zeros(1, self.args.state_size, device=self.args.device)
        self.action = torch.zeros(1, self.args.action_size, device=self.args.device)
        # self.act_history = torch.zeros(1, self.args.action_size*3, device=self.args.device)

        self.speed = 0

        self.step = 0
        self.episode = 0
        self.episode_reward = 0
        self.replay_buffer = []

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

        # self.command_history = np.zeros(3*COMMAND_HISTORY_LENGTH)
        # self.state = np.vstack([image for x in range(FRAME_STACK)])
        self.belief = torch.zeros(1, self.args.belief_size, device=self.args.device)
        self.posterior_state = torch.zeros(1, self.args.state_size, device=self.args.device)
        self.action = torch.zeros(1, self.args.action_size, device=self.args.device)
        # self.act_history = torch.zeros(1, self.args.action_size*3, device=self.args.device)

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
            reward = 1 + (self.speed - self.args.throttle_min) / (self.args.throttle_max - self.args.throttle_min)
            # reward = min(reward, 2) / 2
            # reward = self.speed + 1
            done = self.dead
            reward = reward * -10 if self.dead else reward
            # reward = -self.speed - 10 if self.dead else reward
            # cv2.imwrite("./obs/img_{t}.png".format(t=self.step), self.image)
            next_observation = self.agent.process_im(self.image)

            # self.replay_buffer.append((self.observation,
            #                             self.action.cpu(),
            #                            reward,
            #                             done))

            self.replay_buffer.append((next_observation,
                                                                    self.action.cpu(),
                                                                    reward,
                                                                    done))

            # next_command_history = np.roll(self.command_history, 3)
            # next_command_history[:3] = [self.steering, self.target_speed, self.speed]

            # next_state = np.roll(self.state, 1)
            # next_state[:1, :, :] = self.agent.process_im(self.image, IMAGE_SIZE, RGB)

            # self.replay_buffer.append([ [self.state, self.command_history],
            #                             [self.steering, self.target_speed],
            #                             [reward],
            #                             [next_state, next_command_history],
            #                             [float(not done)]])

            self.episode_reward += reward
            step_end = time.time()

            self.observation = next_observation  # obs is a tensor(3, 64, 64), img is a numpy (120, 180, 3)

            print(
                f"Episode: {self.episode}, Step: {self.step}, Reward: {reward:.2f}, Episode reward: {self.episode_reward:.2f}, Step time: {(self.step_start - step_end):.2f}, Speed: {self.speed:.2f}, Steering, {self.steering:.2f}")

            # self.state = next_state
            # self.command_history = next_command_history

            # print(f"Episode: {self.episode}, Step: {self.step}, Reward: {reward:.2f}, Episode reward: {self.episode_reward:.2f}, Step time: {(self.step_start - step_end):.2f}, Speed: {self.speed:.2f}")

        if self.step > self.args.max_episodes_steps or (self.dead and not self.training):
            self.training_start = time.time()

            self.step = 0
            self.steering = 0
            self.target_speed = 0

            self.training = True
            self.replay_buffer = self.replay_buffer[self.args.skip_initial_steps:]
            return self.steering, self.target_speed, self.training

        if self.training:
            self.training = self.train()
            self.dead = False

            return self.steering, self.target_speed, self.training

        if self.step == 0:
            if not self.sim:
                input("Press Enter to start a new episode.")

            self.reset(self.agent.process_im(self.image))

        self.step += 1

        if self.step < self.args.skip_initial_steps:
            return 0.001, 0, False

        self.step_start = time.time()

        if self.episode <= self.args.random_episodes:
            self.steering = np.random.normal(0, 1)
            self.target_speed = 0.3
            self.action = torch.tensor([[self.steering, self.target_speed]], device=self.args.device)
        else:

            with torch.no_grad():
                self.belief, self.posterior_state = self.agent.infer_state(self.observation.to(self.args.device),
                                                                                                                                        action=self.action,
                                                                                                                                        belief=self.belief,
                                                                                                                                        state=self.posterior_state)
                self.action = self.agent.select_action((self.belief, self.posterior_state))
                # print("before limit", self.action)
                # maintain act_history
                # self.act_history = torch.roll(act_history, -args.action_size, dims=-1)
                # self.act_history[:, -args.action_size:] = action

                # to get steering and target_speed as numpy
                action = self.action.cpu().numpy()  # act dim : [batch_size, act_size]
                # action = self.enforce_limits(action[0], self.steering)  # size [act_size]
                self.steering, self.target_speed = action[0][0], action[0][1]
                # self.action[0] = torch.tensor(action).to(self.action)
                # print("after limit ", self.action)
                ## didn't use enforce_limit yet
                # self.steering, self.target_speed = self.enforce_limits(action, self.command_history[0]) # TODO: change this

        return self.steering, self.target_speed, self.training

        # action = self.agent.select_action((self.state, self.command_history))

        # self.steering, self.target_speed = self.enforce_limits(action, self.command_history[0])

        # return self.steering, self.target_speed, self.training

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

        r = crop[:, :, 0] < threshold
        g = crop[:, :, 1] < threshold
        b = crop[:, :, 2] < threshold

        pixels = (r & g & b).sum()

        # print("Pixels: {}, Required: {}".format(pixels, pixels_required))

        return pixels < pixels_required

    def is_dead_sim(self, img):

        crop_height = 40
        required = 0.8

        cropped = img[-crop_height:]

        rgb = cropped[:, :, 0] > cropped[:, :, 2]

        return rgb.sum() / (crop_height * 160) > required

        
    def enforce_limits(self, action, prev_steering):
        """
        Scale the agent actions to environment limits
        """

        var = (self.args.throttle_max - self.args.throttle_min) / 2
        mu = (self.args.throttle_max + self.args.throttle_min) / 2

        steering_min = max(self.args.steer_limit_left, prev_steering - self.args.max_steering_diff)
        steering_max = min(self.args.steer_limit_right, prev_steering + self.args.max_steering_diff)

        steering = max(steering_min, min(steering_max, action[0]))

        return np.array([steering, action[1] * var + mu], dtype=np.float32)

if __name__ == "__main__":
    #wandb.init(project="dreamer_local")
    print("Starting as training server")
    load_model = args.load_model

    args = define_config()
    #wandb.config.update(args)

    if LOAD_MODEL:
        agent.agent = torch.load(LOAD_MODEL)

    agent = RL_Agent("ari_dreamer", False, args.car_name)
    params_sent = False
    buffer_received = False
    trained = False
    training_episodes = 0
    buffers_received = 0
    prev_buffer = 0
    epi = 0  # add for recording trainig episodes, since the training_episodes seems not useful.

    while training_episodes < args.episodes:
        new_buffer = agent.replay_buffer_sub.run()
        # at beginning, the new_buffer is (0, Ture), when receiving data: (1, data), when training (0, False)
        # print(new_buffer)

        if (new_buffer[0] - 1) == prev_buffer and not trained:
            print("New buffer")
            print(f"{len(new_buffer[1])} new buffer observations")
            #wandb.log({"step": len(new_buffer[1])})
            agent.agent.append_buffer(new_buffer[1])
            prev_buffer += 1
            agent.replay_buffer_received_pub.run(prev_buffer)
            epi += 1

        if new_buffer[1] == False and prev_buffer > 0 and not trained and epi >= args.prefill_episodes:  # add flag to prefill data
            print("Training")
            if not LOAD_MODEL:
                print("Training")
                agent.agent.update_parameters(args.gradient_steps)

            if SAVE_MODEL:
                print("Saving model")
                agent_2 = copy.deepcopy(agent.agent)
                agent_2.replay_buffer.buffer = []
                torch.save(agent_2, MODEL_PATH)

            params = agent.agent.export_parameters()
            trained = True
            print("Sending parameters")
            agent.param_pub.run(params)
            time.sleep(1)

        if new_buffer[1] == False and prev_buffer > 0 and not trained and epi < args.prefill_episodes:  # prefilling data
            print("Prefill random data")
            params = agent.agent.export_parameters()
            trained = True
            agent.param_pub.run(params)
            time.sleep(1)

        if trained and agent.param_sub.run() == False:
            trained = False
            prev_buffer = 0
            print("Waiting for observations.")

    # training_episodes += 1

    time.sleep(0.1)


