import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from DataArgument import transform
from Train_Model import CustomModel

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 학습된 모델 로드
model_path = 'trained_model.pth'
model = CustomModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 이미지 경로 설정
image_folder = 'C:/data/VIS_Onshore/Videos'

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

# 클래스 이름 매핑
class_names = ['X', 'Y', 'Nx', 'Ny']

# 이미지에 경계 상자와 라벨 시각화 함수
def plot_boxes(results, img, class_names):
    img_copy = img.copy()
    for i, coord in enumerate(results):
        x1, y1, nx, ny = coord
        label_str = f"{class_names[0]}: {x1:.2f}, {class_names[1]}: {y1:.2f}, {class_names[2]}: {nx:.2f}, {class_names[3]}: {ny:.2f}"
        img_copy = cv2.putText(img_copy, label_str, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return img_copy

# 이미지 처리 및 시각화
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))

for i, image_file in enumerate(image_files[:16]):
    img_path = os.path.join(image_folder, image_file)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 이미지 변환
    img_transformed = transform(img_rgb).unsqueeze(0).to(device)

    # 예측
    with torch.no_grad():
        results = model(img_transformed).cpu().numpy().reshape(-1, 4)

    # 결과 시각화
    img_with_boxes = plot_boxes(results, img_rgb, class_names)

    # 플롯에 이미지 추가
    ax = axs[i // 4, i % 4]
    ax.imshow(img_with_boxes)
    ax.set_title(image_file)
    ax.axis('off')

plt.tight_layout()
plt.show()