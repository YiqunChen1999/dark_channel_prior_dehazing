
import os, utils, cv2

src_fn = input("Please input path to your source image: ")
while not os.path.exists(src_fn):
    src_fn = input("Directory does not exist, please input again: ")

trg_fn = input("Please input path to your target image: ")
trg_dir = trg_fn.replace(trg_fn.split("/")[-1], "")
if not os.path.exists(trg_dir):
    os.makedirs(trg_dir)

src_img = cv2.imread(src_fn, -1)
trg_img = utils.dcp_dehazing(src_img)
success = cv2.imwrite(trg_fn, trg_img)
if not success:
    print("ERROR: can not save image to {}".format(trg_dir))