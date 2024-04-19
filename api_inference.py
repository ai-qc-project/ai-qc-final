import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from fastapi import FastAPI, File, UploadFile

resnet_model = torch.load("resnet50_model.pth")
resnet_model.eval()
fastercnn_model = torch.load("custom_model.pth")
fastercnn_model.eval()
transform = transforms.Compose([transforms.ToTensor()])

app = FastAPI()

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    image_tensor = transform(image)
    with torch.no_grad():
        resnet_prediction = resnet_model(image_tensor)
    predicted_class_index = torch.argmax(resnet_prediction).item()
    class_labels = ["FAILED", "PASS"]
    predicted_class = class_labels[predicted_class_index]
    if predicted_class != "FAILED":
        return {"message": "Image passed initial classification, skipping Faster R-CNN processing."}
    else:
        with torch.no_grad():
            fastercnn_prediction = fastercnn_model([image_tensor])

        boxes = fastercnn_prediction[0]['boxes']
        labels = fastercnn_prediction[0]['labels']

        output_dir = "api_result"
        os.makedirs(output_dir, exist_ok=True)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_image.save(os.path.join(output_dir, f"cropped_{i}.jpg"))

        return {"message": "Cropped images saved successfully!"}
