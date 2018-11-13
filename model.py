import torch
import torch.nn as nn
import numpy as np
from helper import initialize_weights
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
from params import par


"""
3D Resnet model definition
"""


"""
3D Residual Block definition

Inputs:
feature_maps - number of input and output feature maps for the block
"""
class ResidualBlock(nn.Module):
    def __init__(self, feature_maps):
        super(ResidualBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
                nn.Conv3d(feature_maps, feature_maps, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(feature_maps),
                nn.ReLU(inplace=True),
                nn.Conv3d(feature_maps, feature_maps, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(feature_maps)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        return self.relu(residual + out)


"""
Constructs a level of the Resnet model

Inputs:
num_blocks - number of residual blocks in level
in_feature_maps - number of input feature maps for the first block
    in this level
out_feature_maps - number of feature maps for every other block in level
    and the depth of the output of this level
downsample - whether to learn a residual connection for the first block
    which performs downsampling
"""
class ResidualGroup(nn.Module):
    def __init__(self, num_blocks, in_feature_maps, out_feature_maps, downsample):
        super(ResidualGroup, self).__init__()
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

        if self.downsample:
            self.initial_block = nn.Sequential(
                nn.Conv3d(in_feature_maps, out_feature_maps, kernel_size=3, stride=(1, 2, 2), padding=(2, 2, 2)),
                nn.BatchNorm3d(out_feature_maps)
            )
            self.learned_residual_connection = nn.Sequential(
                    nn.Conv3d(in_feature_maps, out_feature_maps, kernel_size=1, stride=(1, 2, 2), padding=(1, 1, 1)),
                    nn.BatchNorm3d(out_feature_maps)
                )
        else:
            self.initial_block = ResidualBlock(out_feature_maps)
            self.learned_residual_connection = None

        layers = []

        for i in range(1, num_blocks):
            layers.append(ResidualBlock(out_feature_maps))

        self.base = nn.Sequential(*layers)

    def forward(self, x):
        out = self.initial_block(x)

        if self.learned_residual_connection != None:
            residual = self.learned_residual_connection(x)
            out = self.relu(residual + out)

        return self.base(out)


"""
3D Resnet model definition

Inputs:
num_blocks - array of size 4 representing the number of residual blocks
    at each level of the Resnet model
"""
class ResNet3D(nn.Module):
    def __init__(self, num_blocks):
        super(ResNet3D, self).__init__()
        assert(len(num_blocks) == 4)

        #initial convolutional block
        layers = [nn.Sequential(
                nn.Conv3d(par.num_channels, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=3),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
            )]

        #construct residual layers
        for i in range(len(num_blocks)):
            in_feature_maps, out_feature_maps, downsample = 2 ** (i + 5), 2 ** (i + 6), True

            if i == 0:
                in_feature_maps = 64
                downsample = False

            layers.append(ResidualGroup(num_blocks[i], in_feature_maps, out_feature_maps,
                downsample))

        layers.append(nn.AvgPool3d(kernel_size=(3, 3, 3), stride=2, padding=1))
        self.feature_extractor = nn.Sequential(*layers)

        #compute CNN feature extrator output shape
        tmp = Variable(torch.zeros(1, par.num_channels, par.seq_len, par.img_h, par.img_w))
        linear_input_size = int(np.prod(self._extract_features(tmp).size()))

        #define classification head
        self.linear = nn.Sequential(
                nn.Linear(linear_input_size, par.linear_size),
                nn.BatchNorm1d(par.linear_size),
                nn.ReLU(inplace=True),
                nn.Linear(par.linear_size, par.seq_len)
            )

        initialize_weights(self)

    def forward(self, x):
        base_features = self._extract_features(x)
        base_features = base_features.view(base_features.size(0), -1)
        return self.linear(base_features)

    def _extract_features(self, x):
        return self.feature_extractor(x)

    def get_loss(self, x, y):
        pred_speeds = self.predict(x)
        loss = torch.nn.functional.mse_loss(pred_speeds, y.squeeze())
        return loss

    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()

        if par.clip != None:
            torch.nn.utils.clip_grad_norm(self.parameters(), par.clip)

        optimizer.step()
        return loss

    def predict(self, x):
        return self.forward(x).squeeze()


"""
Deep VO model definition
"""


"""
FlowNet convolutional block definition

Inputs:
batch_norm - whether to perform batch normalization
in_feature_maps - number of input feature maps to conv block
out_feature_maps - number of output feature maps from conv block
kernel_size - kernel size used to perform convolution
stride - stride used to perform convolution
dropout - dropout percentage
"""
def conv(batch_norm, in_feature_maps, out_feature_maps, kernel_size=3, stride=1, dropout=0):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_feature_maps, out_feature_maps, kernel_size=kernel_size,
                stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_feature_maps),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_feature_maps, out_feature_maps, kernel_size=kernel_size,
                stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )


"""
Modified DeepVO architecture utilizing base model and new classifier head

Inputs:
h - height of the input frames
w - width of the input frames
batch_norm - whether to perform batch normalization
"""
class DeepVO(nn.Module):
    def __init__(self, h, w, batch_norm=True):
        super(DeepVO,self).__init__()
        self.base_model = BaseDeepVO(h, w, batch_norm)
        self.linear = nn.Linear(in_features=par.rnn_hidden_size, out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        base_features = self.base_model(x)
        out = self.linear(base_features)
        return out

    def get_loss(self, x, y):
        pred_speeds = self.predict(x)
        loss = torch.nn.functional.mse_loss(pred_speeds, y.squeeze())
        return loss

    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()

        if par.clip != None:
            torch.nn.utils.clip_grad_norm(self.rnn.parameters(), par.clip)

        optimizer.step()
        return loss

    def predict(self, x):
        return self.forward(x).squeeze()


"""
Base DeepVO model definition (conv + lstm)

Inputs:
h - height of the input frames
w - width of the input frames
batch_norm - whether to perform batch normalization
"""
class BaseDeepVO(nn.Module):
    def __init__(self, h, w, batch_norm):
        super(BaseDeepVO,self).__init__()
        self.batch_norm = batch_norm

        self.conv1 = conv(self.batch_norm, par.num_channels, 64, kernel_size=7, stride=2, dropout=par.conv_dropout[0])
        self.conv2 = conv(self.batch_norm, 64, 128, kernel_size=5, stride=2, dropout=par.conv_dropout[1])
        self.conv3 = conv(self.batch_norm, 128, 256, kernel_size=5, stride=2, dropout=par.conv_dropout[2])
        self.conv3_1 = conv(self.batch_norm, 256, 256, kernel_size=3, stride=1, dropout=par.conv_dropout[3])
        self.conv4 = conv(self.batch_norm, 256, 512, kernel_size=3, stride=2, dropout=par.conv_dropout[4])
        self.conv4_1 = conv(self.batch_norm, 512, 512, kernel_size=3, stride=1, dropout=par.conv_dropout[5])
        self.conv5 = conv(self.batch_norm, 512, 512, kernel_size=3, stride=2, dropout=par.conv_dropout[6])
        self.conv5_1 = conv(self.batch_norm, 512, 512, kernel_size=3, stride=1, dropout=par.conv_dropout[7])
        self.conv6 = conv(self.batch_norm, 512, 1024, kernel_size=3, stride=2, dropout=par.conv_dropout[8])

        #compute CNN feature extrator output shape
        tmp = Variable(torch.zeros(1, par.num_channels, h, w))
        rnn_input_size = int(np.prod(self._extract_features(tmp).size()))

        self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=par.rnn_hidden_size,
            num_layers=2, dropout=par.rnn_dropout_between, batch_first=True)
        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)

        #initialize layer weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)

                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.LSTM):
                # layer 1
                kaiming_normal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()

                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                kaiming_normal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n//4, n//2
                m.bias_hh_l1.data[start:end].fill_(1.)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.stack_frames(x)
        batch_size = x.size(0)
        seq_len = x.size(1)

        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        x = self._extract_features(x)
        x = x.view(batch_size, seq_len, -1)

        out, hc = self.rnn(x)
        out = self.rnn_drop_out(out)
        return out

    def _extract_features(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

    #stacks frames from input clip into groups of size 2 or 3
    def stack_frames(self, x):
        if par.use_three_frames:
            seq_list = []

            for frame_seq in x:
                frames_list = (frame_seq[0], frame_seq[0], frame_seq[1])
                stacked_frames_list = [torch.cat(frames_list, dim=0).unsqueeze(0)]

                for i in range(1, len(frame_seq) - 1):
                    frames_list = (frame_seq[i-1], frame_seq[i], frame_seq[i+1])
                    stacked_frames_list.append(torch.cat(frames_list, dim=0).unsqueeze(0))

                frames_list = (frame_seq[-2], frame_seq[-1], frame_seq[-1])
                stacked_frames_list.append(torch.cat(frames_list, dim=0).unsqueeze(0))
                seq_list.append(torch.cat(stacked_frames_list, dim=0).unsqueeze(0))

            return torch.cat(seq_list, dim=0)
        else:
            x = torch.cat((x[:, :-1], x[:, 1:]), dim=2)
            return torch.cat((x, x[:, -1].unsqueeze(1)), dim=1)
