import cv2
import numpy as np
import matplotlib.pyplot as plt


folder_path = "../../data/stitching/"

datasets = {
    'boat': ['boat1.jpg', 'boat2.jpg'],
    'budapest': ['budapest1.jpg', 'budapest2.jpg'],
    'newspaper': ['newspaper1.jpg', 'newspaper2.jpg'],
    's': ['s1.jpg', 's2.jpg']
}

def detect_and_match_features(img1, img2, method='SIFT'):
    if method == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create()
    elif method == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
    elif method == 'ORB':
        detector = cv2.ORB_create(nfeatures=1500)

    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

    if descriptors1 is None or descriptors2 is None:
        return None, None, []

    if method == 'ORB':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return keypoints1, keypoints2, good_matches, H
    else:
        return None, None, [], None


def warp_image(img1, img2, H):
    height, width, channels = img2.shape
    warped_image = cv2.warpPerspective(img1, H, (width, height))
    return warped_image


def crop_black_borders(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find all non-black points (the part of the image we want to keep)
    coords = cv2.findNonZero(gray)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped_image = image[y:y + h, x:x + w]
        return cropped_image
    return image


def display_results(dataset_name, img1, img2, results):
    plt.figure(figsize=(15, 10))
    for i, (method, (warped_image, keypoints1, keypoints2, matches)) in enumerate(results):
        plt.subplot(len(results), 3, i * 3 + 1)
        img_keypoints1 = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(cv2.cvtColor(img_keypoints1, cv2.COLOR_BGR2RGB))
        plt.title(f'{method} Image 1 Keypoints')
        plt.axis('off')

        plt.subplot(len(results), 3, i * 3 + 2)
        img_keypoints2 = cv2.drawKeypoints(img2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(cv2.cvtColor(img_keypoints2, cv2.COLOR_BGR2RGB))
        plt.title(f'{method} Image 2 Keypoints')
        plt.axis('off')

        if warped_image is not None:
            cropped_image = crop_black_borders(warped_image)
            plt.subplot(len(results), 3, i * 3 + 3)
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Warped with {method}')
            plt.axis('off')
        else:
            plt.subplot(len(results), 3, i * 3 + 3)
            plt.title(f'Warped with {method} (failed)')
            plt.axis('off')

    plt.tight_layout()
    plt.show()


for dataset_name, filenames in datasets.items():
    img1_path = folder_path + filenames[0]
    img2_path = folder_path + filenames[1]
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print(f"Error loading images for {dataset_name}. Check the file paths.")
        continue

    results = []
    for method in ['SIFT', 'ORB', 'SURF']:
        keypoints1, keypoints2, matches, H = detect_and_match_features(img1, img2, method)
        if H is not None:
            warped_image = warp_image(img1, img2, H)
            results.append((method, (warped_image, keypoints1, keypoints2, matches)))
        else:
            print(f"Not enough matches found using {method} in {dataset_name}.")
            results.append((method, (None, keypoints1, keypoints2, matches)))

    display_results(dataset_name, img1, img2, results)