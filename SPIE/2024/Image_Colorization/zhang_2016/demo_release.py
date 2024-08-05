
import argparse
import matplotlib.pyplot as plt
import os

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
for f in os.listdir(opt.img_path):
	img = load_img(opt.img_path+'/'+f)
	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()

	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())

	plt.imsave(f'{opt.save_prefix}{f}', out_img_eccv16)
	print(f)
