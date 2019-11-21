"""

Inference engine detector sample
 
"""
import sys
import cv2
import argparse
import logging as log
sys.path.append('../src')
from openvino.inference_engine import IENetwork, IECore
from ie_detector import InferenceEngineDetector

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--imagepath', help = 'images folder \
            argument', type = str)
    
    #
    # Add your code here
    #
    
    return parser

def main():
    log.basicConfig(format = "[%(levelname)s] %(message)s", level = log.INFO, 
                               stream = sys.stdout)
    log.info("Poli object detection!")

    detector = InferenceEngineDetector() 
    imagePath = 'C:/UNN_HPC_SCHOOL_2019_ML/Images/dog.jpg' #build_argparse().parse_args()
    image_source = cv2.imread(imagePath)
    #tmp = detector._prepare_image(image_source, h, w)
    tensor = detector.detect(image_source)
    image_final = detector.draw_detection(tensor,image_source)
    #print(tmp)
    cv2.imshow("Original image", image_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return 

if __name__ == '__main__':
    sys.exit(main()) 