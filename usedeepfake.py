import torch
import torchvision.transforms as transforms
import cv2
import PIL
from torch.nn import functional as F

# Assuming DeepfakeVideoDetector is defined elsewhere
from newdeepfakefortest import DeepfakeVideoDetector

# Load saved model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DeepfakeVideoDetector().to(device)
model.load_state_dict(torch.load('best_deepfake_video_detector_try4.pth', map_location=device))
model.eval()  # Set the model to evaluation mode

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def detect_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    seq_len = 16
    frame_count = 0
    
    while cap.isOpened() and frame_count < seq_len:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        frame = PIL.Image.fromarray(frame)
        frame = transform(frame)
        frames.append(frame)
        frame_count += 1
    cap.release()
    
    # Pad frames if necessary
    if len(frames) < seq_len:
        frames += [frames[-1]] * (seq_len - len(frames))
    
    # Convert frames to tensor (B, C, T, H, W)
    tensor = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to(device)
    
    with torch.no_grad():  # Disable gradient computation
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
    
    return predicted.item(), probabilities[0][1].item()  # Return prediction and fake probability

# Example usage
video_path = 'thor.mp4'
result, fake_prob = detect_deepfake(video_path)
print(f'Video {video_path} is {"fake" if result == 1 else "real"} with {fake_prob:.2%} confidence of being fake')