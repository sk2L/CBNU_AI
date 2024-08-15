import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import webbrowser
import tempfile

from CNN_0420_None import Create_CNN

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model_path = 'C:\\CNN_Project\\Saved_models_0420\\best_model_epoch_11.pth'
model = Create_CNN().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 테스트 데이터셋 준비
data_dir = 'C:\\00.캡스톤\\wm_images\\wm_images'
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Train 데이터셋 로드
train_data_dir = os.path.join(data_dir, 'train')
train_data = datasets.ImageFolder(root=train_data_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=False)

# Test 데이터셋 로드
test_data_dir = os.path.join(data_dir, 'test')
test_data = datasets.ImageFolder(root=test_data_dir, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 시각화를 위한 함수들 정의
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def plot_confusion_matrix(cm, classes, ax=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if ax is None:
        ax = plt.gca()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cax = ax.matshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    ax.figure.colorbar(cax, ax=ax)  # 수정된 부분
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()

# Confusion Matrix 계산
def compute_confusion_matrix(model, data_loader, device):
    nb_classes = 9
    confusion_matrix = np.zeros((nb_classes, nb_classes))
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(data_loader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix

# Train 및 Test 데이터셋에 대한 모델 평가
train_confusion_mtx = compute_confusion_matrix(model, train_loader, device)
test_confusion_mtx = compute_confusion_matrix(model, test_loader, device)
class_names = train_data.classes

# 두 데이터셋의 오차 행렬을 시각화하는 함수
def visualize_performance(train_cm, test_cm, classes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    plot_confusion_matrix(train_cm, classes=classes, ax=ax1, normalize=True, title='Train Dataset')
    plot_confusion_matrix(test_cm, classes=classes, ax=ax2, normalize=True, title='Test Dataset')
    plt.show()


def compute_metrics(confusion_mtx):
    # 클래스별로 Precision, Recall, F1-score를 계산합니다.
    precision = np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=0)
    recall = np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)

    precision = np.round(np.nan_to_num(precision) * 100,1)
    recall = np.round(np.nan_to_num(recall) * 100,1)
    f1 = np.round(np.nan_to_num(f1) * 100,1)

    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }

    return metrics


# 메트릭스 계산
train_metrics = compute_metrics(train_confusion_mtx)
test_metrics = compute_metrics(test_confusion_mtx)

# 메트릭스를 DataFrame으로 변환
train_df = pd.DataFrame(train_metrics, index=class_names)
test_df = pd.DataFrame(test_metrics, index=class_names)

# 각 DataFrame의 마지막에 평균값을 추가
train_df.loc['Average'] = train_df.mean()
test_df.loc['Average'] = test_df.mean()

# DataFrame을 화면에 표시하고 포맷팅
def display_formatted_df(df, title=''):
    print(title)
    print(df.applymap("{:.1f}".format))  # DataFrame의 각 요소를 소수 첫째 자리까지의 문자열로 포맷팅하여 표시

# 함수를 사용하여 결과 출력
display_formatted_df(train_df, 'Train Dataset Performance Metrics')
display_formatted_df(test_df, 'Test Dataset Performance Metrics')

def display_in_browser(df, title=''):
    # DataFrame을 HTML로 변환
    formatted_html = df.to_html(float_format="{:.1f}".format)

    # HTML 파일로 저장하기 위해 임시 파일을 생성
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        # HTML 파일에 내용을 쓰고 파일 경로를 저장
        f.write(f'<html><head><title>{title}</title></head><body><h1>{title}</h1>{formatted_html}</body></html>')
        url = f.name

    # 기본 웹 브라우저에서 HTML 파일 열기
    webbrowser.open(url)

# Train 데이터셋의 메트릭스를 브라우저에서 보여줌
display_in_browser(train_df, 'Train Dataset Performance Metrics')

# Test 데이터셋의 메트릭스를 브라우저에서 보여줌
display_in_browser(test_df, 'Test Dataset Performance Metrics')

# 메인 실행 부분
if __name__ == '__main__':
    visualize_performance(train_confusion_mtx, test_confusion_mtx, class_names)
