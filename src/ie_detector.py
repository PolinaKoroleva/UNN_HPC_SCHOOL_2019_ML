"""

Inference engine detector
 
"""
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class InferenceEngineDetector:
    def __init__(self, weightsPath =  'C:/public/mobilenet-ssd/FP32/mobilenet-ssd.bin', configPath =  'C:/public/mobilenet-ssd/FP32/mobilenet-ssd.xml',
                 device_name = 'CPU', extension = 'C:/Program Files (x86)/IntelSWTools/openvino_2019.3.379/inference_engine/bin/intel64/Release/cpu_extension_avx2.dll'):
        self.ie = IECore()
        if extension:
            self.ie.add_extension(extension, 'CPU')
            
        self.net = IENetwork (model=configPath, weights=weightsPath)
         
        self.exec_net = self.ie.load_network(network=self.net, device_name='CPU')
        
        return

    def draw_detection(self, detections, img):
        #print(detections[0,0,1,:].shape)
        #print(detections[0,0,:,:])
        #print(val)
       
        for val in detections[0,0,:,:]:
            print(val[6])
            cv2.rectangle(img, (int(val[3]*img.shape[1]), int(val[4]*img.shape[0])), (int(val[5]*img.shape[1]), int(val[6]*img.shape[0])), (0,255,0), 10)
            cv2.putText(img, str(val[1]), (int(val[3]*img.shape[1]), int(val[4]*img.shape[0]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        return img

    def _prepare_image(self, image, h, w):
        image = cv2.resize(image, dsize = (w, h))
       # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        
        return image
        
    def detect(self, image):
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))
        n, c, h, w = self.net.inputs[input_blob].shape
         
        blob = self._prepare_image(image,h,w)
        
        output = self.exec_net.infer(inputs={input_blob: blob})
        output = output[out_blob]
        
        return output