import datetime
import time
from collections import deque
import numpy as np

class PID(object):

    def __init__(self, p=0.01, i=0.00, d=-0.2):
        self.p = p
        self.d = d
        self.i = i

        self.history_length = 100

        self.derivation_time = 5

        self.history = np.zeros(self.history_length)

        self.throttle = 0.15

        zero = (0, 0, 0)
        self.state = {"a": zero, "v": zero, "x": zero}

    def run(self, vel, pos, acc, target_speed, car_running=True):
        
        try:    
            current_speed = (vel.x**2 + vel.y**2 + vel.z**2)**0.5
            
            self.state = {
                "a": (acc.x, acc.y, acc.z),
                "v": (vel.x, vel.y, vel.z),
                "x": (pos.x, pos.y, pos.z)
            }

        except AttributeError:
            #Measurement from realsense is yet to arrive
            current_speed = 0

        if target_speed == 0 or not car_running:

            self.throttle = 0.15
            return 0, self.state
        
        error = target_speed - current_speed

        self.history = np.roll(self.history, 1)
        self.history[0] = error

        i_error = np.mean(self.history)
        d_error = np.mean(np.diff(self.history[:self.derivation_time]))

        adjustment = self.p * error + self.i * i_error + self.d * d_error

        print("Speed: {:.2f}, Throttle: {:.2f}, Error: {:2f}".format(current_speed, self.throttle, error))
        
        self.throttle += adjustment
        
        return self.throttle, self.state