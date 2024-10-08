import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
import os
import PIL

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

class DeepfakeVideoDetector(nn.Module):
    def __init__(self):
        super(DeepfakeVideoDetector, self).__init__()
        self.cnn = torchvision.models.video.r3d_18(pretrained=True)
        self.cnn.fc = nn.Linear(512, 256)  # Modify the last layer
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)  # Changed to 2 classes: real and fake
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.cnn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class FaceForensicsDataset(Dataset):
    def __init__(self, root_dir, transform=None, seq_len=16):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len
        self.video_paths = []
        self.labels = []
        self.load_dataset()

    def load_dataset(self):
        for dir_path, dir_names, file_names in os.walk(self.root_dir):
            for file_name in file_names:
                if file_name.endswith('.mp4'):
                    video_path = os.path.join(dir_path, file_name)
                    label = 0 if 'real' in video_path else 1  # 0 for real, 1 for fake
                    self.video_paths.append(video_path)
                    self.labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self.load_video(video_path)
        
        if len(frames) < self.seq_len:
            frames = frames + [frames[-1]] * (self.seq_len - len(frames))
        elif len(frames) > self.seq_len:
            step = len(frames) // self.seq_len
            frames = frames[::step][:self.seq_len]
        
        tensor = torch.stack(frames).permute(1, 0, 2, 3)
        return tensor, label

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.seq_len:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = PIL.Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        return frames

transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Reduced size for memory efficiency
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])