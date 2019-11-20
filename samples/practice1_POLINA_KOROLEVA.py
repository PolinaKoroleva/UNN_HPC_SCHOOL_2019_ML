import sys
import cv2
import logging as log
import argparse

sys.path.append('../src')
from imagefilter import ImageFilter

def build_argparse():
    parser=argparse.ArgumentParser()
    
    parser.add_argument('-i', '--image', help = 'your cmd \
            argument', type = str)

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    log.info("Hello image filtering")
    args = build_argparse().parse_args()
    
    imagePath = args.image
    
    log.info(imagePath)
    
    image_source = cv2.imread(imagePath)
    
    log.info(image_source.shape)
    
    myfilter = ImageFilter(gray = True, shape = (100,100))
    
    image_final = myfilter.process_image(image_source)    
    
  # variable = args.argument1    

    
    # Add your code here
    cv2.imshow("Original image", image_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return
    
if __name__ == '__main__':
    sys.exit(main())