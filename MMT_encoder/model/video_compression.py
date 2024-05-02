import cv2, torch
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

# input shape : (batch x channels x frames x height x width)
# output shape : (batch x channels x 3 x (frames-1) x height x width)

#비디오 변환 함수
def calculate_optical_flow(prev_frame, next_frame):
    # 두 프레임을 회색조로 변환합니다.
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Farneback 방법을 사용하여 광학 흐름을 계산합니다.
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def warp_flow(img, flow):   #모션벡터 계산
    # 픽셀의 이동에 따라 이미지를 와핑합니다.
    h, w = flow.shape[:2]
    flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))
    displacement = np.add(flow_map, flow.reshape(-1, 2)).astype(np.float32)

    # 리매핑 함수를 사용하여 와핑된 이미지를 얻습니다.
    warped = cv2.remap(img, displacement[:,0].reshape(h,w), displacement[:,1].reshape(h,w), cv2.INTER_LINEAR)
    return warped

def calculate_residual_frame(next_frame, warped_frame):
    # 잔여 프레임은 현재 프레임과 와핑된 프레임의 차이로 계산됩니다.
    residual = cv2.absdiff(next_frame, warped_frame)
    return residual

def process_frame_pair(prev_frame, next_frame):
    flow = calculate_optical_flow(prev_frame, next_frame)
    warped_frame = warp_flow(prev_frame, flow)
    residual_frame = calculate_residual_frame(next_frame, warped_frame)
    return flow, residual_frame

def video_compression(video_frame):  # tensor (batch x channels x frames x height x width)
    device = video_frame.device  # 입력 텐서의 디바이스를 얻습니다.

    all_flow_tensors = []
    all_residual_tensors = []

    # GPU에서 CUDA 연산 완료 대기
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    for i in range(video_frame.shape[0]):  # video_frame.shape[0]은 batch size입니다.
        video_frame_np = video_frame[i].permute(1, 2, 3, 0).detach().to('cpu').numpy().astype(np.float32)

        flow_tensors = []
        residual_tensors = []

        for j in range(video_frame_np.shape[0] - 1):  # video_frame_np.shape[0]은 frame count입니다.
            # CPU에서 광학 흐름을 계산합니다.
            flow = calculate_optical_flow(video_frame_np[j], video_frame_np[j+1])

            # 모션 벡터를 적용하여 와핑된 프레임을 생성합니다.
            warped_frame = warp_flow(video_frame_np[j], flow)

            # 잔여 프레임을 계산합니다.
            residual_frame = calculate_residual_frame(video_frame_np[j+1], warped_frame)

            # NumPy 배열에서 텐서로 변환합니다.
            flow_tensor = torch.from_numpy(flow).permute(2, 0, 1)
            residual_tensor = torch.from_numpy(residual_frame).permute(2, 0, 1)

            # 변환된 텐서를 입력 텐서와 동일한 디바이스로 이동시킵니다.
            flow_tensors.append(flow_tensor.to(device))
            residual_tensors.append(residual_tensor.to(device))

        # 모든 텐서를 하나의 텐서로 스택합니다.
        all_flow_tensors.append(torch.stack(flow_tensors))
        all_residual_tensors.append(torch.stack(residual_tensors))

    return torch.stack(all_flow_tensors).permute(0, 2, 1, 3, 4), torch.stack(all_residual_tensors).permute(0, 2, 1, 3, 4)
