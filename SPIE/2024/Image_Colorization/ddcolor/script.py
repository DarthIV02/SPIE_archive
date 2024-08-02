import cv2
import os
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img_colorization = pipeline(Tasks.image_colorization, model='damo/cv_ddcolor_image-colorization')
for f in os.listdir('./imagenet_64x64_bw/'):
    result = img_colorization(f'./imagenet_64x64_bw/{f}')
    cv2.imwrite(f'output/{f}', result[OutputKeys.OUTPUT_IMG])
    print(f)