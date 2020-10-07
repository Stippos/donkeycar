import datetime
import os

class EpisodeLogger:

    def __init__(self):
        if not os.path.isdir("./records"):
            os.mkdir("./records")
        
        self.file = f"./records/record_{datetime.datetime.today().isoformat()}.csv"
        
        with open(self.file, "w+") as f:
            f.write("Episode;Step;Time;Steering;Throttle;SpeedX;SpeedY;SpeedZ;PosX;PosY;PosZ\n")

        self.episode = 0
        self.step = 0

    def run(self, training, steering, throttle, pos=None, vel=None):    
        
        if not training:
            
            if self.step == 0:
                self.episode += 1

            self.step += 1

            try:
                location = [pos.x, pos.y, pos.z]
                speed = [vel.x, vel.y, vel.z]
            except AttributeError:
                location = [0, 0, 0]
                speed = [0, 0, 0]
                

            with open(self.file, "a+") as f:
                f.write("{};{};{};{};{};{};{};{};{};{};{}\n".format(
                    self.episode,
                    self.step,
                    datetime.datetime.today().isoformat(),
                    steering,
                    throttle,
                    *speed,
                    *location
                ))
        
        else:
            self.step = 0