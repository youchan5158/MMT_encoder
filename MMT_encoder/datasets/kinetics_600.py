import torchvision
from torchvision import transforms
from torchvision.datasets import Kinetics

# 데이터셋 위치와 파라미터 설정
root = '../../datasets/kinetics-600'  # Kinetics 데이터셋 경로
frames_per_clip = 8  # 한 클립당 프레임 수
step_between_clips = 1  # 클립 간의 프레임 간격

# 데이터 변환 설정 (예: 리사이즈, 텐서 변환, 정규화 등)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 비디오 프레임을 128x128로 리사이즈
    transforms.ToTensor(),  # 이미지 데이터를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# Kinetics 데이터셋 로드
dataset = Kinetics(
    root=root,
    frames_per_clip=frames_per_clip,
    step_between_clips=step_between_clips,
    transform=transform,
    download=False  # 데이터셋 다운로드 필요 유무
)

# 데이터 로더 설정
from torch.utils.data import DataLoader

data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 데이터 로더를 통해 배치 데이터 처리
for i, (video, audio, label) in enumerate(data_loader):
    print(f"Batch {i}:")
    print(f"Video Tensor Shape: {video.shape}")  # 비디오 데이터의 텐서 차원
    print(f"Audio Tensor Shape: {audio.shape}")  # 오디오 데이터의 텐서 차원
    print(f"Label: {label}")  # 레이블 출력
    if i == 1:  # 두 배치만 예시로 처리
        break