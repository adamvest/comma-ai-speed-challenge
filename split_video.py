from skvideo.io import vread
from skimage.io import imsave

train_video = vread("./data/train.mp4")
split_idx = int(train_video.shape[0] * .8)
val_video = train_video[split_idx:]
train_video = train_video[:split_idx]
test_video = vread("./data/test.mp4")

print("Splitting training video...\n")
count = 0

for frame in train_video:
    imsave("./data/train_frames/frame_%d.png" % count, frame)
    count += 1

print("\nSplitting validation video...\n")
count = 0

for frame in val_video:
    imsave("./data/val_frames/frame_%d.png" % count, frame)
    count += 1

print("\nSplitting testing video...\n")
count = 0

for frame in test_video:
    imsave("./data/test_frames/frame_%d.png" % count, frame)
    count += 1

f = open("./data/train.txt", "r")
lines = f.readlines()
f.close()

train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

f = open("./data/train.txt", "w")
f.writelines(train_lines)
f.close()

f = open("./data/val.txt", "w")
f.writelines(val_lines)
f.close()
