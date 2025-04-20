import os
import splitfolders

input_folder = "dataset"
output_folder = "data_split"

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7, .2, .1))
