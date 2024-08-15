import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from DataArgument import load_videos, load_mat_files, CustomDataset, transform, device
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import itertools

# 데이터 로드 및 검증
nir_videos = load_videos('C:/data/NIR', sample_rate=10)  # 프레임 샘플링
nir_data = load_mat_files('C:/data/NIR')

vis_onboard_videos = load_videos('C:/data/VIS_Onboard', sample_rate=10)
vis_onboard_data = load_mat_files('C:/data/VIS_Onboard')

vis_onshore_videos = load_videos('C:/data/VIS_Onshore', sample_rate=10)
vis_onshore_data = load_mat_files('C:/data/VIS_Onshore')

# 데이터셋 및 데이터 로더 설정
dataset = CustomDataset(nir_videos, nir_data, transform=transform, augment=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 모델 정의
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)  # Output: 4 values (X, Y, Nx, Ny)
        )

    def forward(self, x):
        return self.resnet(x)

model = CustomModel().to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 학습 파라미터 설정
num_epochs = 50

# 학습 추이 기록을 위한 리스트
train_losses = []
train_acc = []

# 모델 학습
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.view(-1, 4))  # 레이블을 (batch_size, 4) 형태로 변경
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 예측과 실제 값 비교를 위한 코드 수정
        predicted = outputs.view(-1, 4)
        total += labels.size(0)
        correct += ((predicted - labels.view(-1, 4)).abs() < 0.1).all(dim=1).sum().item()

    scheduler.step()  # 학습률 갱신
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_acc.append(epoch_acc)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

print("Training Finished.")

# 학습된 모델 저장
model_save_path = 'trained_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# 학습 추이 그래프
plt.figure(figsize=(12, 5))

# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_acc, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix 계산 함수
def compute_confusion_matrix(model, dataloader):
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
    return confusion_matrix(all_labels, all_preds), all_labels, all_preds

# 데이터셋 및 데이터 로더 설정 (테스트용)
test_dataset = CustomDataset(vis_onboard_videos, vis_onboard_data, transform=transform, augment=False)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Confusion Matrix 계산
cm, all_labels, all_preds = compute_confusion_matrix(model, test_dataloader)

# Confusion Matrix 시각화
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion Matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# 클래스 이름 정의
class_names = ['X', 'Y', 'Nx', 'Ny']

plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix')

# Precision, Recall, F1-Score 계산
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')