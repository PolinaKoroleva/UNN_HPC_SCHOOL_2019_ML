import cv2

class ImageFilter():
    def __init__(self, gray = False, shape = None, crop = None):
        self.gray = gray
        if shape:
            self.shape = shape
        if crop:
            self.crop = crop
    
    def process_image(self, image):
        #
        # Put your code here, like example
        #
        #if self.shape:
        #    do something
        # 
        #width = image.shape[0] # size of the pic on X-axis
        
        #image[: width,]
        
        if self.shape:
            image = cv2.resize(image, dsize = self.shape)
        
        return image