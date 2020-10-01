class Terminator:

    def __init__(self):
        self.dead = False
        self.training = False
        self.image = None
        self.steering = 0
        self.throttle = 0

    def poll():
        self.dead = self.is_dead(self.image)
        self.steering *= self.dead
        self.throttle *= self.dead

    def update():
        while True:
            self.poll()
    
    def run_threaded(self, image, steering, throttle, training):

    def run(self, image, steering, throttle, training):
        
        return self.run_threaded()

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

        print("Pixels: {}, Required: {}".format(pixels, pixels_required))
        
        return  pixels < pixels_required