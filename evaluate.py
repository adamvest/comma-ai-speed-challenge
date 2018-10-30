import os
import time
import torch
import pandas as pd
from model import DeepVO
from data import get_split_info, VideoClipDataset
from helper import evaluate_predictions, write_predictions
from torch.utils.data import DataLoader
from params import par


split = "val" if par.evaluate_val else "test"
test_df, num_frames = get_split_info(split, par.seq_len, par.overlap)
test_dataset = VideoClipDataset(test_df, par.resize_mode, (par.img_w, par.img_h),
    par.img_means, par.img_stds, par.minus_point_5)
test_dl = DataLoader(test_dataset, batch_size=1, num_workers=par.n_processors,
    pin_memory=par.pin_mem)

print("Number of samples in test dataset: %d\n" % len(test_df.index))

if par.model == "deepvo":
    model = DeepVO(par.img_h, par.img_w, par.batch_norm)
elif par.model == "conv3d":
    model =
else:
    raise ValueError("Invalid model selected")

use_cuda = torch.cuda.is_available()

if use_cuda:
    print("Moving model to GPU...\n")
    model = model.cuda()

if not par.load_weights or not os.path.isfile(par.load_model_path):
	raise ValueError("Pretrained weights not provided!")
else:
    model.load_state_dict(torch.load(par.load_model_path))

print("Begin evaluating model...\n")

running_pred_sums = torch.zeros(num_frames, dtype=torch.float32)
pred_counts = torch.zeros(num_frames, dtype=torch.float32)
total_time = 0.0
model.eval()

for frame_ids, clip, _ in test_dl:
    if use_cuda:
        clip = clip.cuda(non_blocking=par.pin_mem)

    frame_ids = frame_ids.squeeze()
    start_time = time.time()
    predicted_speeds = model.predict(clip).data.cpu()
    total_time += time.time() - start_time

    for i in range(len(frame_ids)):
        running_pred_sums[frame_ids[i]] += predicted_speeds[i]
        pred_counts[frame_ids[i]] += 1

avg_preds = running_pred_sums / pred_counts

if split == "val":
    mse = evaluate_predictions(avg_preds, split)
    print("Average Prediction Time: %.4f" % (total_time / len(test_dl)))
    print("Mean Square Error: %.4f" % float(mse))
else:
    write_predictions(avg_preds, split)
