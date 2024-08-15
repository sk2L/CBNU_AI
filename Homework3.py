import cv2
import numpy as np
import matplotlib.pyplot as plt

folder_path = "../../data/stitching/"

datasets = {
    'boat': ['boat1.jpg', 'boat2.jpg', 'boat3.jpg', 'boat4.jpg', 'boat5.jpg', 'boat6.jpg'],
    'budapest': ['budapest1.jpg', 'budapest2.jpg', 'budapest3.jpg', 'budapest4.jpg', 'budapest5.jpg', 'budapest6.jpg'],
    'newspaper': ['newspaper1.jpg', 'newspaper2.jpg', 'newspaper3.jpg', 'newspaper4.jpg']
}

def create_panorama(images):
    stitcher = cv2.createStitcher()

    stitcher.setPanoConfidenceThresh(0.8)
    stitcher.setSeamEstimationResol(0.01)
    stitcher.setCompositingResol(cv2.Stitcher_ORIG_RESOL)
    stitcher.setWaveCorrection(True)

    status, pano = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        return pano
    else:
        error_messages = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameters adjustment failed"
        }
        error_message = error_messages.get(status, "Unknown error")
        print(f"Error during stitching! Status code: {status} - {error_message}")
        return None

def load_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Error loading image {path}")
        else:
            images.append(img)
    return images

def display_panorama(panorama, dataset_name):
    if panorama is not None:
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.title(f'Panorama for {dataset_name}')
        plt.axis('off')
        plt.show()
    else:
        print(f"Failed to create panorama for {dataset_name}")

for dataset_name, filenames in datasets.items():
    print(f"Processing dataset: {dataset_name}")
    image_paths = [folder_path + filename for filename in filenames]
    images = load_images(image_paths)

    if len(images) < 2:
        print(f"Not enough images to create a panorama for {dataset_name}")
        continue

    print(f"Attempting to create panorama for {dataset_name} with {len(images)} images.")
    panorama = create_panorama(images)
    display_panorama(panorama, dataset_name)


def load_image2(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def display_image(image, title="Image"):
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def visualize_keypoints(image1, image2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    img1_kp = cv2.drawKeypoints(image1, kp1, None, color=(0, 255, 0), flags=0)
    img2_kp = cv2.drawKeypoints(image2, kp2, None, color=(0, 255, 0), flags=0)

    display_image(img1_kp, "Image 1 Keypoints")
    display_image(img2_kp, "Image 2 Keypoints")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    display_image(img_matches, "Matches")

def create_panorama2(images):
    stitcher = cv2.createStitcher()
    status, panorama = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        print("Unable to stitch images, error code =", status)
        return None
    return panorama

def display_panorama2(panorama):
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.title("Panorama Result")
    plt.axis("off")
    plt.show()

image1_path = "../../data/stitching/s1.jpg"
image2_path = "../../data/stitching/s2.jpg"
image1 = load_image2(image1_path)
image2 = load_image2(image2_path)

if image1 is not None and image2 is not None:
    visualize_keypoints(image1, image2)
else:
    print("One or more images failed to load.")

if image1 is not None and image2 is not None:
    images = [image1, image2]
    panorama = create_panorama2(images)
    if panorama is not None:
        display_panorama2(panorama)
    else:
        print("Failed to create panorama.")
else:
    print("One or more images failed to load.")