import sys
import numpy as np

sys.path.insert(1, "~/Documents/RLDonkeyCar/models/")

from ae_sac import AE_SAC

class RL_controller(object):
    def __init__(self):
        self.agent = AE_SAC()
        self.state = 

    def run(self, image, speed, random):
        action = self.agent.select_action()
