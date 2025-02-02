'''
Author: Tawn Kramer
File: realsense2.py
Date: April 14 2019
Notes: Parts to input data from Intel Realsense 2 cameras
'''
import time
import logging

import numpy as np
import pyrealsense2 as rs

class RS_T265(object):
    '''
    The Intel Realsense T265 camera is a device which uses an imu, twin fisheye cameras,
    and an Movidius chip to do sensor fusion and emit a world space coordinate frame that 
    is remarkably consistent.
    '''

    def __init__(self, image_output=False):
        #Using the image_output will grab two image streams from the fisheye cameras but return only one.
        #This can be a bit much for USB2, but you can try it. Docs recommend USB3 connection for this.
        self.image_output = image_output

        # Declare RealSense pipeline, encapsulating the actual device and sensors
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.pose)

        if self.image_output:
            #right now it's required for both streams to be enabled
            cfg.enable_stream(rs.stream.fisheye, 1) # Left camera
            cfg.enable_stream(rs.stream.fisheye, 2) # Right camera

        # Start streaming with requested config
        self.pipe.start(self.cfg)
        self.running = True
        
        zero_vec = (0.0, 0.0, 0.0)
        self.pos = zero_vec
        self.vel = zero_vec
        self.acc = zero_vec
        self.img = None

        self.restarted = False

    def poll(self):
        try:
            if not self.restarted:
                frames = self.pipe.wait_for_frames()
            else:
                return
                
        except Exception as e:
            logging.error(e)
            return

        if self.image_output:
            #We will just get one image for now.
            # Left fisheye camera frame
            left = frames.get_fisheye_frame(1)
            self.img = np.asanyarray(left.get_data())


        # Fetch pose frame
        pose = frames.get_pose_frame()

        if pose:
            data = pose.get_pose_data()
            self.pos = data.translation
            self.vel = data.velocity
            self.acc = data.acceleration

            self.speed = np.sqrt(self.vel.x**2 + self.vel.y**2 + self.vel.z**2)

            logging.debug('realsense pos(%f, %f, %f)' % (self.pos.x, self.pos.y, self.pos.z))

    def update(self):
        while self.running:
            self.poll()

    def run_threaded(self, training):

        if training and not self.restarted and self.speed < 0.05:

            print("Restarting RealSense")
            self.pipe.stop()
            self.restarted = True
                
        if not training and self.restarted:
            print("RealSense restarted")
            self.pipe.start(self.cfg)
            self.restarted = False
        
        return self.pos, self.vel, self.acc, self.img, self.speed

    def run(self, training):
            
        self.poll()

        return self.run_threaded()

    def shutdown(self):
        self.running = False
        time.sleep(0.1)
        self.pipe.stop()


if __name__ == "__main__":
    c = RS_T265()
    while True:
        pos, vel, acc, img = c.run()
        print(pos)
        time.sleep(0.1)
    c.shutdown()
