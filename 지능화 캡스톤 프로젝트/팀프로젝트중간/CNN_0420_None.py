import os
import random
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import datetime
import json
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using {'CUDA' if device.type == 'cuda' else 'CPU'}")

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")

# 데이터 경로 설정
data_dir = 'C:\\00.캡스톤\\wm_images\\wm_images'

# 데이터 증강을 위한 변환 설정
augment_transforms2 = transforms.Compose([
    transforms.Resize((64, 64)),  # 이미지 크기 조정
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
])

# 데이터셋 로딩 시 사용할 변환 (ToTensor 포함)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 이미지 크기 조정
    transforms.ToTensor(),
])

# 텐서를 PIL 이미지로 변환하기 위한 변환 추가
to_pil = ToPILImage()

# 데이터 증강 함수
def augment_images(image_folder, target_count=10000):
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
    current_count = len(images)

    if current_count < target_count:
        print(f"Augmenting images in {image_folder}...")
        while current_count < target_count:
            for img_path in images:
                if current_count >= target_count:
                    break
                image = Image.open(img_path)
                # 이미지를 텐서로 변환하고 증강을 적용
                augmented_image = transform(image)
                # 텐서를 PIL 이미지로 다시 변환
                augmented_image = to_pil(augmented_image)
                # 증강된 이미지 저장
                augmented_image_path = f"{img_path.rsplit('.', 1)[0]}_aug_{current_count}.{img_path.rsplit('.', 1)[-1]}"
                augmented_image.save(augmented_image_path)
                current_count += 1

# 'train' 세트에 대해서만 데이터 증강을 수행
defect_types = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'none', 'Random', 'Scratch']
for defect_type in defect_types:
    print(f"Processing {defect_type} images in train set...")
    image_folder = os.path.join(data_dir, 'train', defect_type)
    augment_images(image_folder)

print("Image count adjustment completed for 'train' set only.")

# 모델 아키텍처 정의
class Create_CNN(nn.Module):
    def __init__(self):
        super(Create_CNN, self).__init__()
        # Conv-Pool-Conv 그룹 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        # Conv-Pool-Conv 그룹 2
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=32)

        # Conv-Pool-Conv 그룹 3
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=64)

        # Conv-Pool-Conv 그룹 4
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(num_features=128)

        self.dropout = nn.Dropout2d(p=0.2)  # Spatial Dropout

        # 완전연결 계층
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=512)
        self.bn_fc1 = nn.BatchNorm1d(num_features=512)

        self.fc2 = nn.Linear(in_features=512, out_features=9)  # 클래스 수에 맞게 조정

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(F.relu(self.bn6(self.conv6(x))))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool4(F.relu(self.bn8(self.conv8(x))))

        x = self.dropout(x)  # Spatial Dropout 적용

        x = x.view(-1, 128 * 4 * 4)  # Flatten the tensor
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)  # Softmax 활성화 함수

# 데이터셋 로딩 및 DataLoader 설정
train_data = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)

test_data = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, drop_last=True)

val_data = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, drop_last=True)

# 모델, 손실 함수 및 최적화 알고리즘 인스턴스화
model = Create_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 학습률 스케줄러 설정

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= total  # 평균 손실 계산
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy


# 모델 저장 경로 설정
model_save_dir = "C:\\CNN_Project\\Saved_models_0420"
os.makedirs(model_save_dir, exist_ok=True)  # 경로가 없다면 생성

# 모델 학습 및 최고 모델 저장 로직
def train_model():
    num_epochs = 50
    best_accuracy = 0.0
    best_model_path = ''
    model.train()
    start_time = datetime.datetime.now()

    # 손실과 정확도를 저장할 리스트 초기화
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):  # 전체 데이터셋을 여러 번(50) 반복
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = datetime.datetime.now()

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 평균 손실과 정확도 계산
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        # 검증 세트를 사용한 모델 평가
        val_loss, val_accuracy = validate(model, test_loader, criterion)

        # 결과 저장
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accuracies.append(epoch_accuracy)
        val_accuracies.append(val_accuracy)

        train_accuracy = 100 * correct / total
        val_loss, val_accuracy = validate(model, val_loader, criterion)

        # 학습 데이터 저장
        with open(training_log_path, 'w') as f:
            json.dump({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
            }, f)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_path = os.path.join(model_save_dir, f'best_model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), best_model_path)  # 모델 저장


        # 시간 계산
        epoch_end = datetime.datetime.now()
        time_elapsed = epoch_end - epoch_start
        total_time_elapsed = epoch_end - start_time
        remaining_epochs = 50 - (epoch + 1)
        average_time_per_epoch = total_time_elapsed / (epoch + 1)
        expected_remaining_time = average_time_per_epoch * remaining_epochs
        expected_end_time = datetime.datetime.now() + expected_remaining_time

        # 에포크 당 결과 로깅
        print(f'Epoch {epoch + 1:2d}/{50} | '
              f'Best {best_accuracy:.2f}% | '
              f'Acc/trn {train_accuracy:.2f}% | '
              f'Acc/val {val_accuracy:.2f}% | '
              f'Time/e {time_elapsed.total_seconds() / 60:.2f}m | '
              f'Elapsed hrs {total_time_elapsed.total_seconds() / 3600:.2f}h | '
              f'Exp.end date {expected_end_time.strftime("%Y-%m-%d %H:%M")} (hrs)')

        scheduler.step()  # 학습률 스케줄러 업데이트

    print(f'Best model saved at: {best_model_path}')
    return train_losses, val_losses, train_accuracies, val_accuracies

# 손실과 정확도를 저장할 파일명을 지정
training_log_path = 'C:\\CNN_Project\\Result\\training_log.json'

# 학습 로그를 파일에 저장하는 함수
def save_training_log(train_losses, val_losses, train_accuracies, val_accuracies, filename='training_log.json'):
    with open(filename, 'w') as log_file:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }, log_file)

# 모델을 평가 모드로 설정 및 평가 실행
def Evaluation() :
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # 각 클래스별로 Precision, Recall, F1-Score 계산
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            all_labels, all_predictions, labels=[i for i in range(len(defect_types))], average=None
        )

        # 데이터프레임 생성
        results_df = pd.DataFrame({
            'No.': range(len(defect_types)),
            'Defect Class': defect_types,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1_score
        })

        # 평균 추가
        average_precision = precision.mean()
        average_recall = recall.mean()
        average_f1_score = f1_score.mean()

        average_df = pd.DataFrame({
            'No.': ['Average'],
            'Defect Class': [''],
            'Precision': [average_precision],
            'Recall': [average_recall],
            'F1-score': [average_f1_score]
        })

        results_df = pd.concat([results_df, average_df], ignore_index=True)

        # 결과를 CSV 파일로 저장
        results_df.to_csv('C:\\CNN_Project\\Result\\evaluation_results.csv', index=False)

    test_loss /= len(test_loader.dataset)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    test_accuracy = 100 * sum(np.array(all_labels) == np.array(all_predictions)) / len(all_labels)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}')

if __name__ == "__main__":
    train_model()
    Evaluation()
