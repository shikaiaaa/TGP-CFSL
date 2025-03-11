import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat, reduce

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def paramsInit(net):
    if isinstance(net, nn.Conv2d):
        nn.init.xavier_uniform_(net.weight.data)
        nn.init.constant_(net.bias.data, 0.0)
    elif isinstance(net, nn.BatchNorm2d):
        net.weight.data.fill_(1)
        net.bias.data.zero_()
    elif isinstance(net, nn.Linear):
        net.weight.data.normal_(0, 0.01)
        net.bias.data.zero_()

class CGConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(CGConvBlock, self).__init__()

        self.in_dim = in_dim
        self.out_dim1 = out_dim

        self.BN = nn.BatchNorm2d(in_dim)
        self.BN2 = nn.BatchNorm2d(in_dim)

        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.conv2d2 = nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=1, stride=1)
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=True),
            nn.BatchNorm3d(out_dim),
        )

        self.block = residual_block(out_dim, out_dim)
        self.block2 = residual_block2(out_dim, out_dim)

        self.ACT = nn.PReLU()
        self.ACT2 = nn.PReLU()

        self.score_linear = nn.Sequential(nn.Linear(out_dim, out_dim, bias=True), nn.Sigmoid())
        self.final_XH = nn.Sequential(nn.Conv2d(out_dim + in_dim, out_dim, kernel_size=1), nn.PReLU())
        paramsInit(self)

    def forward(self, X_in:torch.Tensor, H_in:torch.Tensor):

        b, _, c, h, w = X_in.shape
        _, _, _, n = H_in.shape
        X_in_reshape = X_in.reshape([b, -1, c, h*w])

        X_in_BN = self.BN(X_in_reshape)
        H_in_BN = self.BN2(H_in)

        X = self.conv2d(X_in_BN)
        H = self.conv2d(H_in_BN)

        X_mean = torch.mean(X, dim=-1, keepdim=True)
        X_sim = self.conv2d2(X_mean)
        H_sim = self.conv2d2(H)
        H_sim2 = rearrange(torch.cat([X_sim, H_sim], dim=-1), 'b h w c -> b w h c')
        X_sim = rearrange(X_sim, 'b h w c -> b w c h')
        similarity = torch.sigmoid(torch.matmul(X_sim, H_sim2))  # ai
        similarity = torch.softmax(similarity, dim=-1)  # b*81*5

        X_aggre = torch.matmul(similarity, rearrange(torch.cat([X_mean, H], dim=-1), 'b h w c -> b w c h'))

        score = self.score_linear(X_aggre).permute([0, 3, 1, 2]).unsqueeze(-1)
        X = self.conv3d(X.reshape([b, -1, c, h, w]))

        X = score * X_aggre.reshape([b, -1, c, 1, 1]) + (1 - score) * X

        X = self.ACT(X)
        H = self.ACT2(H)

        H_out = H
        X_out = X
        H_out = self.block2(H_out)
        X_out = self.block(X_out)

        return X_out, H_out

def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer
class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): #(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True) #(batch,8,100,9,9)
        x2 = F.relu(self.conv2(x1), inplace=True) #(batch,8,100,9,9)
        x3 = self.conv3(x2) #(batch,8,100,9,9)

        out = F.relu(x1+x3, inplace=True) #(batch,8,100,9,9)
        return out

def conv2d(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer
class residual_block2(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block2, self).__init__()

        self.conv1 = conv2d(in_channel,out_channel)
        self.conv2 = conv2d(out_channel,out_channel)
        self.conv3 = conv2d(out_channel,out_channel)

    def forward(self, x): #(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True) #(batch,8,100,9,9)
        x2 = F.relu(self.conv2(x1), inplace=True) #(batch,8,100,9,9)
        x3 = self.conv3(x2) #(batch,8,100,9,9)

        out = F.relu(x1+x3, inplace=True) #(batch,8,100,9,9)
        return out