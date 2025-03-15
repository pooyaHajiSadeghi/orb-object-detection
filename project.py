import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_object_orb(query_image, train_image):
    """
    Detect and match key features between two images using ORB (Oriented FAST and Rotated BRIEF).
    
    Parameters:
        query_image (numpy array): The first image (object to detect).
        train_image (numpy array): The second image (scene where the object is located).
    """
    img1 = query_image.copy()
    img2 = train_image.copy()

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Check if descriptors exist to avoid errors
    if descriptors1 is None or descriptors2 is None:
        print("Error: Feature descriptors could not be computed.")
        return

    print(f"Number of features in first image: {descriptors1.shape[0]}")
    print(f"Number of features in second image: {descriptors2.shape[0]}")

    # Draw keypoints on the images
    image_with_keypoints1 = cv2.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0))
    image_with_keypoints2 = cv2.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0))

    # Create BFMatcher object with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance

    print(f"\nTotal matched features: {len(matches)}")

    # Select the top matches to display
    num_matches_to_display = 20
    img_result = cv2.drawMatches(
        query_image, keypoints1,
        train_image, keypoints2,
        matches[:num_matches_to_display],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    print(f"\nDisplaying top {num_matches_to_display} matches.")

    # Show the results
    plt.figure(figsize=[10, 10])
    plt.subplot(231), plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)), plt.title("Query Image")
    plt.subplot(232), plt.imshow(cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)), plt.title("Train Image")
    plt.subplot(233), plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)), plt.title("Matched Features")
    plt.subplot(234), plt.imshow(cv2.cvtColor(image_with_keypoints1, cv2.COLOR_BGR2RGB)), plt.title("Features in Query Image")
    plt.subplot(235), plt.imshow(cv2.cvtColor(image_with_keypoints2, cv2.COLOR_BGR2RGB)), plt.title("Features in Train Image")
    plt.show()

# Load images (Ensure correct paths are provided)
query_img = cv2.imread('img/box.png')
train_img = cv2.imread('img/box_in_scene.png')

# Check if images are loaded correctly
if query_img is None or train_img is None:
    print("Error: One or both images could not be loaded. Check the file paths.")
else:
    find_object_orb(query_img, train_img)
