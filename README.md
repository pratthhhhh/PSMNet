# PSMNet

## Overview
PSMNet (Pyramid Stereo Matching Network) is a stereo depth estimation project designed to accurately compute depth maps from stereo image pairs. This repository contains implementations and experiments related to stereo matching and 3D reconstruction.

## Features
- Pyramid-based stereo matching architecture
- Accurate depth estimation from stereo pairs
- End-to-end trainable neural network
- Support for multiple datasets and benchmarks

## Installation
Clone the repository:
```bash
git clone https://github.com/pratthhhhh/PSMNet.git
cd PSMNet
```

## Usage
Basic usage example:
```python
Load stereo image pair and estimate distance:
```python
from psmnet import PSMNet
import cv2

# Initialize the model
model = PSMNet()

# Load stereo image pair
left_image = cv2.imread('left_image.jpg')
right_image = cv2.imread('right_image.jpg')

# Estimate disparity map and depth
disparity_map = model.estimate_disparity(left_image, right_image)
depth_map = model.disparity_to_depth(disparity_map, baseline=0.15, focal_length=721.5376)

# Distance to specific pixel
distance = depth_map[y, x]
print(f'Distance to object: {distance} meters')
```
```

## Datasets
Supported datasets:
- KITTI
- Middlebury
- ETH3D

## Results
State-of-the-art performance on benchmark datasets.

## References
- Related papers and resources

## License
This project is licensed under the MIT License - see LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit pull requests.

## Contact
For questions or feedback, please open an issue on GitHub.
