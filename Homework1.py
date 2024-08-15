import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Stitching.zip에서 4장의 영상(boat1, budapest1, newspaper1, s1)을 선택한 후에 Canny Edge와 Harris Corner를 검출해서 결과를 출력하는 코드를 작성하시오.

images = ["boat1.jpg", "budapest1.jpg", "newspaper1.jpg", "s1.jpg"]
folder_path = "../../data/stitching"

def feature_detection(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded. Check the file path.")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_img, 100, 200)

    harris_corners = np.copy(gray_img)
    harris = cv2.cornerHarris(gray_img, 2, 3, 0.04)
    harris_corners[harris > 0.01 * harris.max()] = 255

    return img, edges, harris_corners


fig, axs = plt.subplots(len(images), 3, figsize=(15, 10))

for i, img_name in enumerate(images):
    image_path = f"{folder_path}/{img_name}"
    try:
        original_img, edges, harris_corners = feature_detection(image_path)
    except ValueError as e:
        print(e)
        continue

    image_title = img_name[:-4]
    fig.text(0.05, 0.85 - i * 0.25, image_title, fontsize=12, ha='right', va='center', rotation='vertical')

    axs[i, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axs[i, 0].set_title('Original Image')
    axs[i, 0].axis('off')

    axs[i, 1].imshow(edges, cmap='gray')
    axs[i, 1].set_title('Canny Edges')
    axs[i, 1].axis('off')

    axs[i, 2].imshow(harris_corners, cmap='gray')
    axs[i, 2].set_title('Harris Corners')
    axs[i, 2].axis('off')

plt.tight_layout()
plt.show()