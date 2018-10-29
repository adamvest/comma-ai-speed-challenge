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

            if np.isnan(flow).any():
                raise ValueError("nan in flow")

            if np.isinf(flow).any():
                raise ValueError("inf in flow")

            mag, ang = cv2.cartToPolar(flow[:, :,0], flow[:, :,1])
            mag = np.clip(mag, 0, np.inf)

            if np.isnan(ang).any():
                raise ValueError("nan in ang")
            if np.isinf(ang).any():
                raise ValueError("inf in ang")

            if np.isnan(mag).any():
                raise ValueError("nan in mag before norm")
            if np.isinf(mag).any():
                print(np.max(mag))
                print(np.min(mag))
                raise ValueError("inf in mag before norm")

            mag2 = cv2.normalize(mag, dst=None,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            if np.isnan(mag2).any():
                raise ValueError("nan in mag after norm")
            if np.isinf(mag2).any():
                raise ValueError("inf in mag after norm")

            flow = np.stack((mag, ang))
            np.save("./data/optical_flow/%s_frames/frame_%d" % (split, i), flow)
