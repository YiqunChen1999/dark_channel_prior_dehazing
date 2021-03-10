
import os, utils

src_dir = input("Please input your source directory: ")
while not os.path.exists(src_dir):
    src_dir = input("Directory does not exist, please input again: ")

trg_dir = input("Please input your target directory: ")
if not os.path.exists(trg_dir):
    os.makedirs(trg_dir)

utils.dcp_dehazing_from_dir(src_dir, trg_dir)

