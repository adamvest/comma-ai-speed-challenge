import os
import time
import torch
import pandas as pd
from model import DeepVO, ResNet3D
from data import get_split_info, VideoClipDataset
from helper import evaluate_predictions, write_predictions
from torch.utils.data import DataLoader
from params import par


#load dataset
split = "val" if par.evaluate_val else "test"
test_df, num_frames = get_split_info(split, par.seq_len, par.overlap)
test_dataset = VideoClipDataset(test_df, par.resize_mode, (par.img_w, par.img_h),
    par.img_means, par.img_stds, par.minus_point_5)
test_dl = DataLoader(test_dataset, batch_size=1, num_workers=par.n_processors,
    pin_memory=par.pin_mem)

print("Number of samples in test dataset: %d\n" % len(test_df.index))

#create model
if par.model == "deepvo":
    model = DeepVO(par.img_h, par.img_w, par.batch_norm)
elif par.model == "resnet3d":
	if par.resnet_depth == 18:
		num_blocks = [2, 2, 2, 2]
	elif par.resnet_depth == 34:
		num_blocks = [3, 4, 6, 3]
	else:
		raise NotImplementedError("Invalid choice of Resnet depth!")

	model = ResNet3D(num_blocks)
else:
    raise NotImplementedError("Invalid model selected!")

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

#process clips
for frame_ids, clip, _ in test_dl:
    if use_cuda:
        clip = clip.cuda(non_blocking=par.pin_mem)

    frame_ids = frame_ids.squeeze()
    start_time = time.time()
    predicted_speeds = model.predict(clip).data.cpu()
    total_time += time.time() - start_time

    #update prediction averages based on frame ids
    for i in range(len(frame_ids)):
        running_pred_sums[frame_ids[i]] += predicted_speeds[i]
        pred_counts[frame_ids[i]] += 1

running_pred_sums[-1], running_pred_sums[-2] = running_pred_sums[-3], running_pred_sums[-3]
pred_counts[-1], pred_counts[-2] = 1, 1
avg_preds = (running_pred_sums / pred_counts)

#calcuate evaluation metrics
if split == "val":
    mse = evaluate_predictions(avg_preds, split)
    print("Average Prediction Time: %.4f" % (total_time / len(test_dl)))
    print("Mean Square Error: %.4f" % float(mse))

#write test predictions to file
else:
    print("Saving predictions to file...")
    write_predictions(avg_preds, split)
