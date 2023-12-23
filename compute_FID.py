import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
import os

#########################################################################################################################
# In this script we use the PyTorch FID_score function, we compute the FID between the original dataset and the samples #
#########################################################################################################################

def calculate_fid(relative_path1, relative_path2):
    # Get the absolute paths based on the current working directory
    base_path = os.getcwd()
    path1 = os.path.join(base_path, relative_path1)
    path2 = os.path.join(base_path, relative_path2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fid_value = calculate_fid_given_paths([path1, path2],
                                          batch_size=100,
                                          device=device,
                                          dims=2048)
    return fid_value

if __name__ == '__main__':
    # Define the relative paths to the image folders. The folders must contain single image files
    relative_path_to_folder1 = 'exp/image_samples/Car_sigmoid_samples'
    relative_path_to_folder2 = 'exp/datasets/car'

    # Calculate FID
    fid_score = calculate_fid(relative_path_to_folder1, relative_path_to_folder2)
    print(f"FID score: {fid_score}")
