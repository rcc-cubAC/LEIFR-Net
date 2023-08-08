import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import re
class CustomDataset(Dataset):
    def __init__(self, data_root):
        """
        Args:
            data (List or Tensor): A list of your data or a PyTorch tensor. 
            labels (List or Tensor): Corresponding labels for each entry in data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataroot = data_root
        self.lightroot = os.path.join(self.dataroot,'params')
        self.sceneroot = os.path.join(self.dataroot,'scene_pair')
        self.albedoroot = os.path.join(self.dataroot,'albedo_pair')
        self.keypointroot = os.path.join(self.dataroot,'keypoint')
        self.synfaceroot = os.path.join(self.dataroot,'syn_face')
        self.dictroot = os.path.join(self.dataroot,'scene_pair')
        self.dictlst = os.listdir(self.dictroot)
        self.dictlst.sort()
        self.transform = transforms.Compose(
            (
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            )
        )
    

    def split_filename(self,filename):
        # 正则表达式匹配 'flipped' 和 'lightmap_x' (其中x可以是任意数字)
        match = re.match(r'(lightmap_\d+_flipped|lightmap_\d+)_(.*\.png)', filename)
        if match:
            pair = match.groups()
            return pair[0]+'.png',pair[1]
        else:
            return None
    def __len__(self):

        return len(self.dictlst)

    def __getitem__(self, idx):
        filename = self.dictlst[idx]
        lightname,facename=self.split_filename(filename)
        

        # 加载light npz 文件
        lightpath = os.path.join(self.lightroot,('param_'+lightname[9:-3]+'npz').replace('_flipped',''))
        
        loaded_npz = np.load(lightpath)
        loaded_dict = {key: loaded_npz[key] for key in loaded_npz}
        
        light_sources = torch.Tensor(loaded_dict['light_sources']).reshape(2,)
        intensity = torch.Tensor(loaded_dict['intensity']).reshape(1,)
        delay = torch.Tensor(loaded_dict['decay_factor']).reshape(1,)
        
        light = torch.cat([light_sources,intensity,delay],dim=0)

        scenepath = os.path.join(self.sceneroot,filename)
        scene = Image.open(scenepath).convert('RGB')
        scene = self.transform(scene)

        albedopath = os.path.join(self.albedoroot,filename)
        albedo = Image.open(albedopath).convert('RGB')
        albedo = self.transform(albedo)

        keypointpath = os.path.join(self.keypointroot,filename[:-3]+'npy')
        keypoint = torch.Tensor(np.load(keypointpath))
        
        synfacepath = os.path.join(self.synfaceroot,filename)
        synface = Image.open(synfacepath).convert('RGB')
        synface = self.transform(synface)

        return light,scene,albedo,keypoint,synface

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataset = CustomDataset(data_root='E:/Relight/dataset')
    train_loader = DataLoader(
        dataset=CustomDataset(data_root='E:/Relight/dataset'),
        batch_size = 8,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    pbar = tqdm(train_loader)
    for step,(light,scene,albedo,keypoint,synface) in enumerate(pbar):
        print(light.shape,scene.shape,albedo.shape,keypoint.shape,synface.shape)
        