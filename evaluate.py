import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from models import stackhourglass, basic
from dataset import get_dataloaders

def plot_disparity_static(imgL, imgR, disp_true, model, focal_length, baseline, cuda_available):
    model.eval()
    if cuda_available:
        imgL = imgL.cuda()
        imgR = imgR.cuda()

    with torch.no_grad():
        pred_disp = model(imgL, imgR)

    pred_disp = pred_disp.squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Show Left Image
    axes[0].imshow(imgL.squeeze().permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Left Image")
    axes[0].axis('off')

    # Show Ground Truth Disparity
    axes[1].imshow(disp_true.squeeze().cpu().numpy(), cmap='plasma')
    axes[1].set_title("Ground Truth Disparity")
    axes[1].axis('off')
    
    # Show Predicted Disparity
    im = axes[2].imshow(pred_disp, cmap='plasma')
    axes[2].set_title("Predicted Disparity with Distance Legend")
    axes[2].axis('off')
    
    # Get the min and max disparity from the prediction to set the colorbar range
    min_disp = np.percentile(pred_disp[pred_disp > 0], 5) if len(pred_disp[pred_disp > 0]) > 0 else 0
    max_disp = np.percentile(pred_disp, 95) if len(pred_disp) > 0 else 1
    
    cbar = fig.colorbar(im, ax=axes[2], orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label('Disparity Value')
    ticks = cbar.get_ticks()
    
    # Calculate the corresponding distances and create new labels
    distance_labels = []
    for disp_val in ticks:
        if disp_val > 0:
            distance = (focal_length * baseline) / disp_val
            distance_labels.append(f'{distance:.1f}m')
        else:
            distance_labels.append('Inf') # Infinite distance for zero disparity
            
    cbar.ax.set_xticklabels(distance_labels)

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='PSMNet Evaluation')
    parser.add_argument('--datapath', default='/mnt/c/Users/prath/OneDrive/Desktop/Assignments/Thesis/Dataset/DrivingStereo', help='dataset path')
    parser.add_argument('--model_name', default='stackhourglass', choices=['stackhourglass', 'basic'], help='model chosen')
    parser.add_argument('--loadmodel', default='./best_checkpoint.tar', help='load model path')
    args = parser.parse_args()

    maxdisp = 192
    cuda_available = torch.cuda.is_available()
    FOCAL_LENGTH = 721 
    BASELINE = 0.54 

    if args.model_name == 'stackhourglass':
        model = stackhourglass(maxdisp)
    elif args.model_name == 'basic':
        model = basic(maxdisp)
    else:
        raise ValueError('Model not recognized')
        
    if cuda_available:
        model = nn.DataParallel(model)
        model.cuda()
        
    try:
        state_dict = torch.load(args.loadmodel, map_location='cuda' if cuda_available else 'cpu', weights_only=True)
        model.load_state_dict(state_dict['state_dict'])
    except Exception as e:
        print(f"Could not load checkpoint: {e}")

    _, TestImgLoader = get_dataloaders(args.datapath)

    # Test random evaluation plot
    imgL, imgR, disp_L = next(iter(TestImgLoader))
    rand_idx = random.randint(0, imgL.size(0) - 1)
    
    print("Plotting results for a random test image...")
    plot_disparity_static(
        imgL[rand_idx].unsqueeze(0), 
        imgR[rand_idx].unsqueeze(0), 
        disp_L[rand_idx].unsqueeze(0), 
        model,
        focal_length=FOCAL_LENGTH,
        baseline=BASELINE,
        cuda_available=cuda_available
    )

if __name__ == '__main__':
    main()
