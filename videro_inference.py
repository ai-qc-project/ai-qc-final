import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2

resnet_model = torch.load("resnet.pth")
resnet_model.eval()
fastercnn_model = torch.load("faster_cnn.pth")
fastercnn_model.eval()
transform = transforms.Compose([transforms.ToTensor()])
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
image_tensor = transform(frame_pil)

with torch.no_grad():
    resnet_prediction = resnet_model(image_tensor)

predicted_class_index = torch.argmax(resnet_prediction).item()
class_labels = ["FAILED", "PASS"]
predicted_class = class_labels[predicted_class_index]
if predicted_class != "FAILED":
    print("Image passed initial classification, skipping Faster R-CNN processing.")
else:
    with torch.no_grad():
        fastercnn_prediction = fastercnn_model([image_tensor])
    boxes = fastercnn_prediction[0]['boxes']
    labels = fastercnn_prediction[0]['labels']
    output_dir = "video_result"
    os.makedirs(output_dir, exist_ok=True)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
        cv2.imwrite(os.path.join(output_dir, f"cropped_{i}.jpg"), cropped_image)

    print("Cropped images saved successfully!")

cap.release()
cv2.destroyAllWindows()
