import os
import time
import torch
import numpy as np
import pandas as pd
from model import DeepVO
from data_helper import get_split_info, RandomBatchSampler, VideoClipDataset
from torch.utils.data import DataLoader
from params import par


if os.path.isfile(par.train_data_info_path) and os.path.isfile(par.valid_data_info_path):
	print("Load data info from %s" % par.train_data_info_path)
	train_df = pd.read_pickle(par.train_data_info_path)
	val_df = pd.read_pickle(par.valid_data_info_path)
else:
	print("Create new data info")
	train_df = get_split_info("train", par.seq_len, par.overlap)
	val_df = get_split_info("val", par.seq_len, par.overlap)
	train_df.to_pickle(par.train_data_info_path)
	val_df.to_pickle(par.valid_data_info_path)

train_sampler = RandomBatchSampler(train_df, par.batch_size, drop_last=True)
train_dataset = VideoClipDataset(train_df, par.resize_mode, (par.img_w, par.img_h),
    par.img_means, par.img_stds, par.minus_point_5)
train_dl = DataLoader(train_dataset, batch_sampler=train_sampler,
    num_workers=par.n_processors, pin_memory=par.pin_mem)

val_sampler = RandomBatchSampler(val_df, par.batch_size, drop_last=True)
val_dataset = VideoClipDataset(val_df, par.resize_mode, (par.img_w, par.img_h),
    par.img_means, par.img_stds, par.minus_point_5)
val_dl = DataLoader(val_dataset, batch_sampler=val_sampler,
    num_workers=par.n_processors, pin_memory=par.pin_mem)

print("Number of samples in training dataset: %d" % len(train_df.index))
print("Number of samples in validation dataset: %d\n" % len(val_df.index))

deep_vo = DeepVO(par.img_h, par.img_w, par.batch_norm)
use_cuda = torch.cuda.is_available()

if use_cuda:
    print("Moving model to GPU...\n")
    deep_vo = deep_vo.cuda()

if par.optim["opt"] == "Adam":
	optimizer = torch.optim.Adam(deep_vo.parameters(), lr=0.001, betas=(0.9, 0.999))
elif par.optim["opt"] == "Adagrad":
	optimizer = torch.optim.Adagrad(deep_vo.parameters(), lr=par.optim["lr"])

if par.resume:
	pretrained_dict = torch.load(par.load_model_path)

	if par.load_base_deepvo:
		model_dict = deep_vo.base_model.state_dict()
		exclude, strict = [], True

		if par.img_w != 608 or par.img_h != 184:
			exclude.append("rnn.weight_ih_l0")
			strict = False

		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict
			and k not in exclude}
		deep_vo.base_model.load_state_dict(pretrained_dict, strict=strict)
	else:
		deep_vo.load_state_dict(pretrained_dict)

	print("Loaded weights from: %s\n" % par.load_model_path)

print("Begin training model...\n")

min_train_loss, min_val_loss = np.inf, np.inf

for epoch in range(par.epochs):
	start_time = time.time()
	deep_vo.train()
	losses, total_loss = [], 0

	for batch_num, (_, clip, speeds) in enumerate(train_dl):
		if use_cuda:
			clip = clip.cuda(non_blocking=par.pin_mem)
			speeds = speeds.cuda(non_blocking=par.pin_mem)

		loss = deep_vo.step(clip, speeds, optimizer).data.cpu().numpy()
		losses.append(float(loss))
		total_loss += float(loss)

		if par.batch_updates:
			print("Batch Number %d Loss: %.3f" % (batch_num + 1, float(loss)))

	stop_time = time.time()
	epoch_time = stop_time - start_time
	mins = epoch_time // 60
	secs = int(epoch_time - (mins * 60))
	loss_mean = total_loss / len(train_dl)
	print("\nEpoch %d Training Time: %d minutes %d seconds" % (epoch + 1, mins, secs))
	print("Mean MSE Loss: %.3f\n" % loss_mean)

	start_time = time.time()
	deep_vo.eval()
	val_losses, total_val_loss = [], 0

	for _, clip, speeds in val_dl:
		if use_cuda:
			clip = clip.cuda(non_blocking=par.pin_mem)
			speeds = speeds.cuda(non_blocking=par.pin_mem)

		val_loss = deep_vo.get_loss(clip, speeds).data.cpu().numpy()
		val_losses.append(float(val_loss))
		total_val_loss += float(val_loss)

	stop_time = time.time()
	val_time = stop_time - start_time
	val_mins = val_time // 60
	val_secs = int(val_time - (val_mins * 60))
	val_loss_mean = total_val_loss / len(val_dl)
	print("Epoch %d Validation Time: %d minutes %d seconds" % (epoch + 1, val_mins, val_secs))
	print("Mean MSE Loss: %.3f\n" % val_loss_mean)

	if val_loss_mean < min_val_loss and epoch % par.check_interval == 0:
		min_val_loss = val_loss_mean
		print("Save model at epoch %d, mean of valid loss: %.3f" % (epoch + 1, val_loss_mean))
		torch.save(deep_vo.state_dict(), par.save_model_path + ".val_%d" % epoch)

	if loss_mean < min_train_loss and epoch % par.check_interval == 0:
		min_train_loss = loss_mean
		print("Save model at epoch %d, mean of train loss: %.3f" % (epoch + 1, loss_mean))
		torch.save(deep_vo.state_dict(), par.save_model_path + ".train_%d" % epoch)
