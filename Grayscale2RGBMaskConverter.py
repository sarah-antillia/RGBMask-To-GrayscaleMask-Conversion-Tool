# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Grayscale2RGBMaskConverter.py

# 1 Read a mutli-class-rgb-mask file.
# 2 Split the mutli-class-rgb-mask to some multi-class-grayscale-masks.
# 3 Merge the grayscale-masks to one grayscale mask. 
# 4 Save the merged grayscale-masks to a file.

# https://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity


"""
On the grascaling parameters to convert a set of (r,g,b) color pixles to a grayscale intensity,
please, see also: 

https://en.wikipedia.org/wiki/Luma_(video)

For digital formats following CCIR 601 (i.e. most digital standard definition formats), 
luma is calculated with this formula:

gray = 0.299 * R + 0.587 * G + 0.114 * B

Formats following ITU-R Recommendation BT. 709 (i.e. most digital high definition formats)  
use a different formula:

gray = 0.2126 * R + 0.7152 * G + 0.0722 * B
"""

import os
import sys
import cv2
import glob
import numpy as np
import shutil
import traceback

from ConfigParser import ConfigParser

class Grayscale2RGBMaskConverter:
  # Section name in a config_file.
  RGB_CONVERTER = "grayscale2rgb_converter"

  def __init__(self, config_file):
    self.config      = ConfigParser(config_file)
    self.config.dump_all()
    self.masks_dir    = self.config.get(self.RGB_CONVERTER, "masks_dir")
    self.output_dir   = self.config.get(self.RGB_CONVERTER, "output_dir")
    self.class_names  = self.config.get(self.RGB_CONVERTER, "class_names")
    self.mask_colors  = self.config.get(self.RGB_CONVERTER, "mask_colors")
    self.image_width  = self.config.get(self.RGB_CONVERTER, "image_width")
    self.image_height = self.config.get(self.RGB_CONVERTER, "image_height")
    self.num_classes  = self.config.get(self.RGB_CONVERTER, "num_classes")

    self.mask_channels= self.config.get(self.RGB_CONVERTER, "mask_channels")
    self.mask_colors  = self.config.get(self.RGB_CONVERTER, "mask_colors")
    self.gray_mergin  = self.config.get(self.RGB_CONVERTER, "gray_mergin", dvalue=10)
    # (R, G, B) intensity for converting a rgb to a grayscale: CCIR 601
    self.grayscaling = self.config.get(self.RGB_CONVERTER, "grayscaling",
                                        dvalue=(0.299, 0.587, 0.114))

    self.color_order = self.config.get(self.RGB_CONVERTER, "color_order", dvalue="bgr")

    # Preparation: create a gray_map from self.grayscaling and self.mask_colors 
    self.create_gray_map()

    if os.path.exists(self.output_dir):
      shutil.rmtree(self.output_dir)
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

  def convert(self):
    mask_files  = glob.glob(self.masks_dir + "/*.png")
    mask_files += glob.glob(self.masks_dir + "/*.jpg")    
    for mask_file in mask_files:
      self.convert_one(mask_file)

  def colorize_mask_one(self, mask, color=(255, 255, 255), gray=0):
    h, w = mask.shape[:2]
    rgb_mask = np.zeros((w, h, 3), np.uint8)
    condition = (mask[...] == gray) 
    rgb_mask[condition] = [color]    

    return rgb_mask
   
  def create_gray_map(self):
     print("=== create_gray_map")
     self.gray_map = []
     (IR, IG, IB) = self.grayscaling
     for color in self.mask_colors:
       (b, g, r) = color
       if self.color_order == "rgb":
         (r, g, b) = color
       gray = int(IR* r + IG * g + IB * b)
       self.gray_map += [gray]
     print("--- created gray_map {}".format(self.gray_map))
  
  def convert_one(self, mask_file):
    print("=== convert_one {}".format(mask_file))
    mask = cv2.imread(mask_file)
    h, w = mask.shape[:2]
    basename = os.path.basename(mask_file)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Create an empty rgb (3-channels) background
    merged_rgb_mask = np.zeros((w, h, 3), np.uint8)
    for i in range(len(self.mask_colors)):
        color = self.mask_colors[i]
        gray  = self.gray_map[i]

        rgb_mask = self.colorize_mask_one(mask, color=color, gray=gray)
        # Overlapping each mask rgb to the background 
        merged_rgb_mask += rgb_mask
    output_filepath = os.path.join(self.output_dir, basename)
    cv2.imwrite(output_filepath, merged_rgb_mask )
    print("--- Saved {}".format(output_filepath))   

if __name__ == "__main__":
  try:
    config_file = "./grayscale2rgb_converter.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found configuration file {}".format(config_file))
    
    converter = Grayscale2RGBMaskConverter(config_file)
    converter.convert()

  except:
    traceback.print_exc()