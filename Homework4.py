import cv2
import numpy as np
import matplotlib.pyplot as plt

image1_path = "../../data/stitching/dog_a.jpg"
image2_path = "../../data/stitching/dog_b.jpg"
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7)

lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

good_new = p1[st == 1]
good_old = p0[st == 1]

mask = np.zeros_like(img1)

for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = int(new[0]), int(new[1])
    c, d = int(old[0]), int(old[1])
    mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
    mask = cv2.circle(mask, (a, b), 5, (0, 255, 0), -1)

output = cv2.addWeighted(img2, 0.7, mask, 0.3, 0)

plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Optical Flow using Pyramid Lucas-Kanade')
plt.axis('off')
plt.show()