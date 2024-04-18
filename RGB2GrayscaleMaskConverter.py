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

# RGB2GrayscaleMaskConverter.py

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

class RGB2GrayscaleMaskConverter:
  # Section name in a config_file.
  GRAYSCALE_CONVERTER = "grayscale_converter"
  PREPROCESS          = "preprocess"

  def __init__(self, config_file):
    self.config      = ConfigParser(config_file)
    self.config.dump_all()
    self.masks_dir   = self.config.get(self.GRAYSCALE_CONVERTER, "masks_dir")
    self.output_dir  = self.config.get(self.GRAYSCALE_CONVERTER, "output_dir")
    self.class_names = self.config.get(self.GRAYSCALE_CONVERTER, "class_names")
    self.mask_colors = self.config.get(self.GRAYSCALE_CONVERTER, "mask_colors")

    # Preprocessing experiment to adjust the original rgb_mask to a better rgb_mask. 
    self.preprocess  = self.config.get(self.GRAYSCALE_CONVERTER, self.PREPROCESS, dvalue=False)
    self.alpha       = self.config.get(self.PREPROCESS, "alpha", dvalue=3.0)
    self.beta        = self.config.get(self.PREPROCESS, "beta",  dvalue=0)

    # (R, G, B) intensity for converting rgb to grayscale: CCIR 601
    self.grayscaling = self.config.get(self.GRAYSCALE_CONVERTER, "grayscaling", dvalue=(0.299, 0.587, 0.114))

    self.color_order = self.config.get(self.GRAYSCALE_CONVERTER, "color_order", dvalue="bgr")
    if os.path.exists(self.output_dir):
      shutil.rmtree(self.output_dir)
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

  def convert(self):
    mask_files  = glob.glob(self.masks_dir + "/*.png")
    mask_files  += glob.glob(self.masks_dir + "/*.jpg")
    for mask_file in mask_files:
      self.convert_one(mask_file)

  def adjust(self, mask, alpha=1.0, beta=0.0):
    mask = alpha * mask + beta
    return np.clip(mask, 0, 255).astype(np.uint8)

  def convert_one(self, mask_file):
    print("=== convert_one {}".format(mask_file))
    mask = cv2.imread(mask_file)
    basename = os.path.basename(mask_file)
    # Experimental
    if self.preprocess:
      mask = self.adjust(mask, alpha=3.0, beta=0.0)
  
    h, w = mask.shape[:2]
    # create an empty grayscale_mask
    merged_grayscale_mask = np.zeros((w, h, 1), np.uint8)

    for color in self.mask_colors:
      grayscale_mask = self.create_categorized_grayscale_mask(mask, color)
      merged_grayscale_mask += grayscale_mask

    output_filepath = os.path.join(self.output_dir, basename)
    cv2.imwrite(output_filepath, merged_grayscale_mask )
    #print("--- merged_grayscale_mask {}".format(merged_grayscale_mask.shape))
    print("--- Saved {}".format(output_filepath))
      
  def create_categorized_grayscale_mask(self, mask, color):
    (h, w, c) = (0, 0, 0)
    if len(mask.shape) == 3:
      h, w, c = mask.shape[:3]

    # create a grayscale 1 channel black background 
    grayscale_mask = np.zeros((w, h, 1), np.uint8)
    # bgr
    (b, g, r) = color
    condition = (mask[..., 0] == b) & (mask[..., 1] == g) & (mask[..., 2] == r)
    if self.color_order == "rgb":
      (r, b, g) = color
      condition = (mask[..., 0] == r) & (mask[..., 1] == g) & (mask[..., 2] == b)

    # https://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity
    # See also: 
    # https://en.wikipedia.org/wiki/Luma_(video)
    """
    For digital formats following CCIR 601 (i.e. most digital standard definition formats), 
    luma is calculated with this formula:
    gray = 0.299 * R + 0.587 * G + 0.114 * B

    Formats following ITU-R Recommendation BT. 709 (i.e. most digital high definition formats)  
    use a different formula:
    gray = 0.2126 * R + 0.7152 * G + 0.0722 * B
    """
    # CCIR 601
    (IR, IG, IB) = self.grayscaling
    gray = int(IR * r + IG * g + IB * b)
    # BT. 709
    #gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)

    grayscale_mask[condition] = [gray]    
    return grayscale_mask


if __name__ == "__main__":
  try:
    config_file = "./grayscale_converter.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found configuration file {}".format(config_file))
    converter = RGB2GrayscaleMaskConverter(config_file)
    converter.convert()

  except:
    traceback.print_exc()