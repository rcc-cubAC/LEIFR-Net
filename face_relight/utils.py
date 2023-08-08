import torch
from torchvision.utils import save_image

def denormalize(tensor):
    mean = 0.5
    std = 0.5
    tensor = tensor.clone()  # Clone the tensor to not make changes in-place
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)  # Clamping the values to [0,1] just in case they are out of this range
    return tensor

def show(tensors, filename):
    # tensors is a list of tensors
    # filename is the name of the file to save the images to

    denormalized_tensors = [denormalize(t) for t in tensors]

    concatenated_images = torch.cat(denormalized_tensors, dim=2)
    
    # Move the tensor to cpu before saving
    concatenated_images = concatenated_images.cpu()

    save_image(concatenated_images, filename, nrow=1)

