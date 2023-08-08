import torch
import torch.nn as nn
from block import *
import torch.nn.functional as F
import numpy as np

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, layers_channels, up_block_types,down_block_types):
        super(UNet, self).__init__()
        
        self.layers_channels = layers_channels
        
        # Encoding path
        self.encoders = nn.ModuleList()
        for out_chs,name in zip(layers_channels,down_block_types):
            self.encoders.append(get_block(name,in_channels, out_chs))
            in_channels = out_chs

        self.mid = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(0.2),
                    )
        # Decoding path
        self.decoders = nn.ModuleList()
        
        for out_chs,name in zip(reversed(layers_channels[:-1]),up_block_types):
            
            self.decoders.append(get_block(name,in_channels*2, out_chs))
            in_channels = out_chs
            
        self.decoders.append(get_block(up_block_types[-1],in_channels*2, out_channels))

    
    def forward(self, x):
        # Encoding path
        skips = []
        for encoder in self.encoders:
            
            x = encoder(x)
            skips.append(x)
        
        x = self.mid(x)
        # Decoding path
        for decoder, skip in zip(self.decoders, reversed(skips)):
            
            x = decoder(torch.cat([x, skip], 1))

        return x

class PIFu(nn.Module):
    def __init__(self,latent_dim):
        super(PIFu,self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim,latent_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim,latent_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim,latent_dim//2),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim//2,3),
            nn.Tanh()
        )

    def compute_distance(self,feature_map,keypoint):
        # 获取feature_map的大小
        b, c1, h, w = feature_map.shape
        keypoint = keypoint*feature_map.shape[2]
        # 创建一个与feature_map同样大小的grid
        y_grid, x_grid = torch.meshgrid(torch.arange(0, h), torch.arange(0, w),indexing='xy')
        y_grid = y_grid[None, None, :, :].float().to(feature_map.device)
        x_grid = x_grid[None, None, :, :].float().to(feature_map.device)

        # keypoint shape: b, num, 2
        y_key = keypoint[:, :, 0][:, :, None, None]
        x_key = keypoint[:, :, 1][:, :, None, None]

        # 计算距离
        dist = ((x_key - x_grid)**2 + (y_key - y_grid)**2).sqrt()

        return dist

    def concatenate_features(self,feature_map,global_lightcode,keypoint):
        b, c1, h, w = feature_map.shape
        _, c2 = global_lightcode.shape

        # 计算距离
        dist = self.compute_distance(feature_map, keypoint) # shape: b, num, h, w

        # 扩展globalcode维度以匹配feature_map
        expanded_globalcode = global_lightcode[:, :, None, None].expand(b, c2, h, w)

        # 连接feature_map, globalcode和距离
        concatenated = torch.cat([feature_map, expanded_globalcode, dist], dim=1)
        
        return concatenated

    
    def forward(self,feature_map,global_lightcode,keypoint):
        cat_feature = self.concatenate_features(feature_map, global_lightcode, keypoint)
        b, c1, h, w = feature_map.shape
        _, c2 = global_lightcode.shape
        _, c3, _ = keypoint.shape
        cat_feature = cat_feature.permute(0, 2, 3, 1)
        # 将输入调整为(b*h*w, c)
        cat_feature = cat_feature.contiguous().view(-1, c1+c2+c3)
        # 通过MLP
        out = self.mlp(cat_feature)
        # 还原形状为(b, 3, h, w)
        out = out.contiguous().view(b, h, w, -1).permute(0, 3, 1, 2)
        return out

class  Pipeline(nn.Module):
    def __init__(self,light_dim=4,img_dim=6,light_code_dim=32,key_point_num=68,feature_dim=156):
        super(Pipeline,self).__init__()
        down_block_types = ['ResDownBlock2D','ResDownBlock2D','ResDownBlock2D','ResDownBlock2D']
        up_block_types = ['ResUpBlock2D','ResUpBlock2D','ResUpBlock2D','ResUpBlock2D']
        self.unet = UNet(
            in_channels=img_dim,
            out_channels=feature_dim,
            layers_channels=[64, 128, 256, 256],
            down_block_types=down_block_types,
            up_block_types=up_block_types
            )
        self.pifu = PIFu(latent_dim=key_point_num+light_code_dim+feature_dim)
        self.light_encoder = nn.Sequential(
            nn.Linear(light_dim,16),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
            nn.Linear(16,32),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
            nn.Linear(32,light_code_dim),
        )
    
    def forward(self,light,scene,face,keypoint):
        feature_map = self.unet(torch.cat([face,scene],dim=1))
        global_lightcode = self.light_encoder(light)
        out = self.pifu(feature_map,global_lightcode,keypoint)
        
        return out
    
if __name__ == '__main__':
    import torch.optim as optim
    batch_size=8
    img_size=256
    light = torch.rand(batch_size,4).cuda()
    scene = torch.rand(batch_size,3,img_size,img_size).cuda()
    face = torch.rand(batch_size,3,img_size,img_size).cuda()
    keypoint = torch.rand(batch_size,68,2).cuda()
    relight_face = torch.rand(batch_size,3,img_size,img_size).cuda()
    pipe = Pipeline().cuda()
    opt = optim.Adam(pipe.parameters(),lr=1e-5)
    while True:
        
        out = pipe(light,scene,face,keypoint)
        loss = torch.sum(torch.abs(out-relight_face))
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)