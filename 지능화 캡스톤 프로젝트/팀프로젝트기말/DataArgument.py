import os
import cv2
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random
import time

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")


# 파일 경로 설정
base_dir = 'C:/data'
nir_dir = os.path.join(base_dir, 'NIR')
vis_onboard_dir = os.path.join(base_dir, 'VIS_Onboard')
vis_onshore_dir = os.path.join(base_dir, 'VIS_Onshore')
sub_dirs = ['HorizonGT', 'ObjectGT', 'TrackGT', 'Videos']

def load_videos(directory, sample_rate=10):
    videos = {}
    video_dir = os.path.join(directory, 'Videos')
    print(f"Attempting to load videos from {video_dir}...")
    for file_name in os.listdir(video_dir):
        if file_name.endswith('.avi'):
            video_path = os.path.join(video_dir, file_name)
            if not os.path.isfile(video_path):
                print(f"Video file {video_path} does not exist.")
                continue
            print(f"Found video file: {video_path}")
            cap = cv2.VideoCapture(video_path)
            frames = []
            start_time = time.time()
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % sample_rate == 0:
                    frames.append(frame)
                frame_count += 1
            cap.release()
            end_time = time.time()
            duration = end_time - start_time
            videos[file_name] = frames
            print(f"Loaded {len(frames)} frames from {video_path} in {duration:.2f} seconds")
    return videos

def load_mat_files(directory, preferred_keys=['structXML', 'Track']):
    mat_data = {}
    for subdir in sub_dirs[:-1]:  # 'Videos' 디렉토리는 제외
        full_dir = os.path.join(directory, subdir)
        print(f"Attempting to load .mat files from {full_dir}...")
        for file_name in os.listdir(full_dir):
            if file_name.endswith('.mat'):
                file_path = os.path.join(full_dir, file_name)
                if not os.path.isfile(file_path):
                    print(f".mat file {file_path} does not exist.")
                    continue
                print(f"Found .mat file: {file_path}")
                mat_contents = sio.loadmat(file_path)
                key_found = False
                for key in preferred_keys:
                    if key in mat_contents:
                        mat_data[file_name] = mat_contents[key]
                        key_found = True
                        break
                if not key_found:
                    print(f"Preferred keys not found in {file_path}, available keys: {list(mat_contents.keys())}")
    return mat_data

# 데이터 로드 및 검증
nir_videos = load_videos(nir_dir, sample_rate=10)  # 프레임 샘플링
nir_data = load_mat_files(nir_dir)

vis_onboard_videos = load_videos(vis_onboard_dir, sample_rate=10)
vis_onboard_data = load_mat_files(vis_onboard_dir)

vis_onshore_videos = load_videos(vis_onshore_dir, sample_rate=10)
vis_onshore_data = load_mat_files(vis_onshore_dir)

# Transformations: PIL 이미지 변환은 __getitem__에서 적용
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class CustomDataset(Dataset):
    def __init__(self, video_frames, mat_files, transform=None, augment=True):
        self.frames = video_frames
        self.labels = mat_files
        self.transform = transform
        self.augment = augment
        self.all_frames = [frame for video in self.frames.values() for frame in video]
        self.all_labels = []

        # 프레임 수에 맞게 레이블을 확장
        for video_name, frames in self.frames.items():
            base_name = os.path.splitext(video_name)[0]
            corresponding_mat_file = [key for key in self.labels.keys() if base_name in key]
            if corresponding_mat_file:
                num_frames = len(frames)
                label_data = self.labels[corresponding_mat_file[0]]
                if isinstance(label_data, np.ndarray):
                    labels = []
                    for row in label_data:
                        labels.append([row['X'][0][0], row['Y'][0][0], row['Nx'][0][0], row['Ny'][0][0]])
                    labels = np.array(labels)  # 리스트를 numpy 배열로 변환
                    self.all_labels.extend([labels] * num_frames)

        # 레이블의 수가 프레임의 수와 동일한지 확인
        print(f"Total frames: {len(self.all_frames)}, Total labels: {len(self.all_labels)}")

    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, idx):
        if idx >= len(self.all_frames) or idx >= len(self.all_labels):
            raise IndexError("Index out of range")
        image = Image.fromarray(self.all_frames[idx])
        label = self.all_labels[idx]

        if self.augment:
            image = self.apply_augmentation(image)

        if self.transform:
            image = self.transform(image)

        return image.to(device), torch.tensor(label, dtype=torch.float32).to(device)

    def apply_augmentation(self, image):
        if random.random() > 0.5:
            return self.copy_paste(image)
        else:
            return self.mix_up(image)

    def copy_paste(self, image):
        other_idx = random.randint(0, len(self.all_frames) - 1)
        other_image = Image.fromarray(self.all_frames[other_idx])

        patch_size = (50, 50)
        x, y = random.randint(0, other_image.width - patch_size[0]), random.randint(0, other_image.height - patch_size[1])
        patch = other_image.crop((x, y, x + patch_size[0], y + patch_size[1]))

        x, y = random.randint(0, image.width - patch_size[0]), random.randint(0, image.height - patch_size[1])
        image.paste(patch, (x, y))

        return image

    def mix_up(self, image):
        other_idx = random.randint(0, len(self.all_frames) - 1)
        other_image = Image.fromarray(self.all_frames[other_idx])

        alpha = random.uniform(0.4, 0.6)
        mixed_image = Image.blend(image, other_image, alpha)

        return mixed_image

# 데이터셋 및 데이터 로더 확인
dataset = CustomDataset(nir_videos, nir_data, transform=transform, augment=True)
print(f"Dataset size: {len(dataset)}")

if len(dataset) > 0:
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for i, (images, labels) in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print(f"Images shape: {images.shape}, Labels: {labels}")
        if i == 0:  # 첫 번째 배치만 출력 후 중단
            break
else:
    print("No data available in the dataset.")


# 증강된 데이터 개수 확인
augmented_dataset = CustomDataset(nir_videos, nir_data, transform=transform, augment=True)
augmented_dataloader = DataLoader(augmented_dataset, batch_size=10, shuffle=True)

num_augmented_samples = 0
for i, (images, labels) in enumerate(augmented_dataloader):
    num_augmented_samples += len(images)

print(f"Number of augmented samples: {num_augmented_samples}")


__all__ = ['CustomDataset', 'load_videos', 'load_mat_files', 'nir_dir', 'transform']