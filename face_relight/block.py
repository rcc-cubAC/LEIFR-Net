import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Shape of x: [batch_size, channels, height, width]
        batch_size, channels, height, width = x.size()
        
        # Query, key, and value
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)
        # print(q.shape,k.shape)
        # Attention weights
        attention = self.softmax(torch.bmm(q, k))

        # Output
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        return out + x
    
class AttnUpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels,num_layers = 2):
        super(AttnUpBlock2D, self).__init__()
        
        # Upsampling
        self.up_sample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
        
        # Residual Block
        self.reslayer = nn.ModuleList()
        for i in range(num_layers):
            self.reslayer.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                    )
                )

        

        # Self-Attention
        self.self_attention = SelfAttention(out_channels)
        self.out = nn.LeakyReLU(0.2)
    def forward(self, x):
        # Upsampling
        out = self.up_sample(x)
        
        # Residual Connection
        identity = out
        
        # Residual Block
        for block in self.reslayer:
            out = block(out)
        
        
        # Self-Attention
        out = self.self_attention(out)
        
        # Add residual connection
        out += identity
        out = self.out(out)
        
        return out
    

class ResUpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels,num_layers = 2):
        super(ResUpBlock2D, self).__init__()
        
        # Upsampling
        self.up_sample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
        
        # Residual Block
        self.reslayer = nn.ModuleList()
        for i in range(num_layers):
            self.reslayer.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                    )
                )

        
        self.out = nn.LeakyReLU(0.2)
    def forward(self, x):
        # Upsampling
        out = self.up_sample(x)
        
        # Residual Connection
        identity = out
        
        # Residual Block
        for block in self.reslayer:
            out = block(out)
        
        
        # Add residual connection
        out += identity
        out = self.out(out)
        
        return out
    


class AttnDownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels,num_layers = 2):
        super(AttnDownBlock2D, self).__init__()
        
        # Downsampling
        self.down_sample = nn.Sequential(
            
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
        
        # Residual Block
        self.reslayer = nn.ModuleList()
        for i in range(num_layers):
            self.reslayer.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                    )
                )

        

        # Self-Attention
        self.self_attention = SelfAttention(out_channels)
        self.out = nn.LeakyReLU(0.2)
    def forward(self, x):
        # Upsampling
        out = self.down_sample(x)
        
        # Residual Connection
        identity = out
        
        # Residual Block
        for block in self.reslayer:
            out = block(out)
        
        
        # Self-Attention
        out = self.self_attention(out)
        
        # Add residual connection
        out += identity
        out = self.out(out)
        
        return out
    


class ResDownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels,num_layers = 2):
        super(ResDownBlock2D, self).__init__()
        
        # Downsampling
        self.down_sample = nn.Sequential(
            
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
        
        # Residual Block
        self.reslayer = nn.ModuleList()
        for i in range(num_layers):
            self.reslayer.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                    )
                )

        
        self.out = nn.LeakyReLU(0.2)
    def forward(self, x):
        # Down_sampling
        out = self.down_sample(x)
        
        # Residual Connection
        identity = out
        
        # Residual Block
        for block in self.reslayer:
            out = block(out)
        
        
        # Add residual connection
        out += identity
        out = self.out(out)
        
        return out
    
def get_block(name,in_channels,out_channels,num_layers=2):
    
    if name == 'ResDownBlock2D':
        return ResDownBlock2D(in_channels,out_channels,num_layers)
    
    elif name == 'AttnDownBlock2D':
        return AttnDownBlock2D(in_channels,out_channels,num_layers)
    
    elif name == 'ResUpBlock2D':
        return ResUpBlock2D(in_channels,out_channels,num_layers)
    
    elif name == 'AttnUpBlock2D':
        return AttnUpBlock2D(in_channels,out_channels,num_layers)