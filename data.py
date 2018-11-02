import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from params import par


"""
Creates a pandas dataframe containing clip information

Inputs:
split - split name in ["train", "test", "val"]
seq_len - length of the clip in frames
overlap - number of shared frames between clips

Outputs:
data_info - dataframe containing clip information
"""
def get_split_info(split, seq_len, overlap):
    assert(overlap < seq_len)
    assert split in ["train", "test", "val"]
    ids, img_paths, flow_paths, speeds = [], [], [], []

    img_fpaths = glob.glob("./data/%s_frames/*.png" % split)
    flow_fpaths = glob.glob("./data/optical_flow/%s_frames/*.npy" % split)
    img_fpaths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
    flow_fpaths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))

    if split != "test":
        speeds_file = open("./data/%s.txt" % split)
        speed_annotations = speeds_file.readlines()
        speed_annotations = [float(val) for val in speed_annotations]
        speeds_file.close()
    else:
        speed_annotations = [0] * len(img_fpaths)

    for i in range(0, len(img_fpaths)-seq_len+1, seq_len-overlap):
        ids.append([i for i in range(i, i+seq_len)])
        img_paths.append(img_fpaths[i:i+seq_len])
        speeds.append(speed_annotations[i:i+seq_len])
        flow_path_seq = flow_fpaths[i:i+seq_len]

        #for last clip in video, reuse last optical flow frame as
        #we cannot compute the flow for the final frame in the video
        if len(flow_path_seq) < seq_len:
            flow_path_seq.append(flow_path_seq[-1])
            assert(len(flow_path_seq) == seq_len)

        flow_paths.append(flow_path_seq)

    data = {"frame_ids": ids, "image_paths": img_paths, "flow_paths": flow_paths,
        "speeds": speeds}
    return pd.DataFrame(data, columns=list(data.keys())), len(img_fpaths)


"""
Pytorch random batch sampler

Inputs:
info_dataframe - pandas dataframe used to create dataset
batch_size - number of clips per batch
drop_last - whether to keep "incomplete" final batch
"""
class RandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last=False):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last

        num_samples = len(self.df.index)
        num_batches = num_samples // self.batch_size

        if not self.drop_last and num_samples % self.batch_size != 0:
            num_batches += 1

        self.len = num_batches

    def __iter__(self):
        num_samples = len(self.df.index)
        rand_idxs = torch.randperm(num_samples).tolist()
        list_batch_indexes = [rand_idxs[s * self.batch_size:(s * self.batch_size) + self.batch_size]
            for s in range(0, self.len)]
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len


"""
Pytorch dataset for video clips

Inputs:
info_dataframe - pandas dataframe used to create dataset
resize_mode - how to perform resize operation, center crop or interpolation
new_size - resulting size of the frames after resizing
img_means - mean of image channels used for normalization
img_stds - std of image channels used for normalization
minus_point_5 - whether to normalize to range [-.5, .5] instead of [0, 1]
"""
class VideoClipDataset(Dataset):
    def __init__(self, info_dataframe, resize_mode="crop", new_size=None, img_means=None, img_std=(1,1,1), minus_point_5=False):
        img_ops, flow_ops = [], []

        if resize_mode == "crop":
            img_ops.append(transforms.CenterCrop((new_size[0], new_size[1])))
            flow_ops.append(NumpyCenterCrop((new_size[0], new_size[1])))
        elif resize_mode == "rescale":
            img_ops.append(transforms.Resize((new_size[0], new_size[1])))

            if par.use_both or par.use_optical_flow:
                raise ValueError("Cannot resize optical flow image")

        img_ops.append(transforms.ToTensor())
        flow_ops.append(NumpyToTensor())
        self.img_transformer = transforms.Compose(img_ops)
        self.flow_transformer = transforms.Compose(flow_ops)
        self.minus_point_5 = minus_point_5

        if self.minus_point_5:
            self.normalizer = transforms.Normalize(mean=img_means[0], std=img_std)
        else:
            self.normalizer = transforms.Normalize(mean=img_means[1], std=img_std)

        self.data_info = info_dataframe
        self.frame_ids = np.asarray(self.data_info.frame_ids)
        self.image_arr = np.asarray(self.data_info.image_paths)
        self.flow_arr = np.asarray(self.data_info.flow_paths)
        self.speeds_arr = np.asarray(self.data_info.speeds)

    def __getitem__(self, idx):
        frame_ids_seq = torch.tensor(self.frame_ids[idx])
        image_path_seq = self.image_arr[idx]
        flow_path_seq = self.flow_arr[idx]
        speed_seq = torch.FloatTensor(self.speeds_arr[idx])
        image_seq, flow_seq = [], []

        if par.use_both or not par.use_optical_flow:
            for img_path in image_path_seq:
                img = Image.open(img_path)
                img = self.img_transformer(img)

                if self.minus_point_5:
                    img -= 0.5

                img = self.normalizer(img)
                img = img.unsqueeze(0)
                image_seq.append(img)

        if par.use_both or par.use_optical_flow:
            for flow_path in flow_path_seq:
                flow = np.load(flow_path)
                flow = self.flow_transformer(flow).unsqueeze(0)
                flow_seq.append(flow)

        if par.use_both:
            frame_seq = [torch.cat([image_seq[i], flow_seq[i]], 1)
                for i in range(par.seq_len)]
            clip = torch.cat(frame_seq, 0)
        elif par.use_optical_flow:
            clip = torch.cat(flow_seq, 0)
        else:
            clip = torch.cat(image_seq, 0)

        if par.model == "resnet3d":
            clip = clip.permute(1, 0, 2, 3)

        return (frame_ids_seq, clip, speed_seq)

    def __len__(self):
        return len(self.data_info.index)


"""
Custom transform to apply center crop to a Numpy ndarray

Inputs:
crop_size - tuple (w, h) to crop the image to
"""
class NumpyCenterCrop():
    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.delta_x = crop_size[0] // 2
        self.delta_y = crop_size[1] // 2

    def __call__(self, arr):
        _, w, h = arr.shape
        start_x = (w // 2) - self.delta_x
        start_y = (h // 2) - self.delta_y
        return arr[:, start_x:start_x+self.crop_size[0],
            start_y:start_y+self.crop_size[1]]


"""
Custom transform to cast Numpy ndarray to Pytorch Tensor
"""
class NumpyToTensor():
    def __init__(self):
        pass

    def __call__(self, arr):
        return torch.from_numpy(arr)
