from torch import nn
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch
import argparse
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import Pipeline
from utils import show 
import os
def init_args():
    parser = argparse.ArgumentParser(description="A simple argument parser")


    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch_num', type=int, default=10000)
    parser.add_argument('--data_root', type=str, default='E:/Relight/dataset')


    # 解析参数
    args = parser.parse_args()

    return args

def train(args):
    
    pipe = Pipeline().cuda()
    opt = optim.Adam(pipe.parameters(),lr = 1e-5)
    train_loader = DataLoader(
        dataset=CustomDataset(args.data_root),
        batch_size = args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    for epoch in range(args.epoch_num):
        epoch_loss = 0
        pbar = tqdm(train_loader)
        for step,(light,scene,albedo,keypoint,synface) in enumerate(pbar):
            light = light.cuda()
            albedo = albedo.cuda()
            scene = scene.cuda()
            keypoint = keypoint.cuda()
            synface = synface.cuda()
            relightface = pipe(light,scene,albedo,keypoint)
            
            loss = torch.abs(relightface-synface).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss
            pbar.set_postfix(loss='%.6f'%(epoch_loss/(step+1)))
        show([albedo[0], scene[0], synface[0], relightface[0]], os.path.join('./result/output_epoch{}.png'.format(epoch)))
        torch.save(pipe,'./model.pth')
def main():
    args = init_args()
    train(args)

    
if __name__ == '__main__':
    main()
    