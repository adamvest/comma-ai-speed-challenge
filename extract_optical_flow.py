import os
import cv2
import numpy as np


for split in ["train", "test", "val"]:
    print("Extracting optical flow from %s images\n" % split)

    for root, _, fnames in os.walk("./data/%s_frames" % split):
        fnames.sort()

        for i in range(len(fnames) - 1):
            img1 = cv2.imread(os.path.join(root, fnames[i]), flags=cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(os.path.join(root, fnames[i + 1]), flags=cv2.IMREAD_GRAYSCALE)
            flow = cv2.calcOpticalFlowFarneback(img1, img2, flow=None, pyr_scale=.5,
                levels=3, winsize=13, iterations=3, poly_n=5, poly_sigma=1.1,
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            mag, ang = cv2.cartToPolar(flow[:, :,0], flow[:, :,1])
            mag = np.clip(mag, 0, np.inf)
            mag = cv2.normalize(mag, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            flow = np.stack((mag, ang))
            np.save("./data/optical_flow/%s_frames/frame_%d" % (split, i), flow)
