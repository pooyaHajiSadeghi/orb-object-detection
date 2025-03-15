# SIFT & ORB Feature Matching

## Overview
This project demonstrates object detection and feature matching using **ORB (Oriented FAST and Rotated BRIEF)**. The code extracts keypoints from images, matches them, and visualizes the results using OpenCV and Matplotlib.

## Features
- Detects key features in images using ORB
- Matches features between a query image and a train image
- Displays feature points and matched keypoints
- Uses brute-force matching for descriptor comparison

## Installation
Ensure you have Python and the required libraries installed.

```bash
pip install numpy opencv-python matplotlib
```

## Usage
1. Place your images in the `images/` directory.
2. Update the file paths in the script:
    ```python
    query_image = cv2.imread('images/query.png')
    train_image = cv2.imread('images/train.png')
    ```
3. Run the script:
    ```bash
    python feature_matching.py
    ```

## Example Output
After running the script, you should see:
- Keypoints detected in both images
- Matched keypoints visualized
- Console output displaying the number of detected and matched features

## Requirements
- Python 3.7+
- OpenCV
- NumPy
- Matplotlib

## Contributing
Feel free to submit pull requests or report issues.

## License
This project is open-source and available under the MIT License.

## Contact
For questions, reach out via GitHub issues or email.

---

**Note:** Replace `query.png` and `train.png` with your actual image paths. 🚀

