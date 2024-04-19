import shutil
import os
import random
import torch
import torchvision
from PIL import Image

folder = 'Dataset/test'

try:
    shutil.rmtree('Dataset/test')
except:
    print("Testing Dir Does Not Exist")
    
class_names = ['PASS', 'FAIL']
root_dir = 'Dataset'
source_dirs = ['PASS', 'FAIL']

"Dataset/test"

if os.path.isdir(os.path.join(root_dir, source_dirs[1])):
    os.mkdir(os.path.join(root_dir, 'test'))
    
    for i, d in enumerate(source_dirs):
        os.rename(os.path.join(root_dir, d), os.path.join(root_dir, class_names[i]))

    for c in class_names:
        os.mkdir(os.path.join(root_dir, 'test', c))

    for c in class_names:
        images = [x for x in os.listdir(os.path.join(root_dir, c)) if x.lower().endswith('jpg') or x.lower().endswith('jpeg')]
        selected_images = random.sample(images, 20)
        for image in selected_images:
            source_path = os.path.join(root_dir, c, image)
            target_path = os.path.join(root_dir, 'test', c, image)
            shutil.copy2(source_path, target_path)
            
class DatasetProcess(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x[-3:].lower().endswith('jpg') or x.lower().endswith('jpeg')]
            print("Found the images list")
            return images
        self.images = {}
        self.class_names = ["PASS", "FAIL"]
        
        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)
        
        self.image_dirs = image_dirs
        self.transform = transform
    
    def __len__(self):
        return sum(len(self.images[class_name]) for class_name in self.class_names)
    
    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)

    
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(255, 255)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(255, 255)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dirs= {
    "PASS": "Dataset/PASS",
    "FAIL": "Dataset/FAIL"
}

test_dirs= {
    "PASS": "Dataset/test/PASS",
    "FAIL": "Dataset/test/FAIL"
}

train_dataset = DatasetProcess(train_dirs, train_transform)
test_dataset = DatasetProcess(test_dirs, test_transform)


dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)

images, labels = next(iter(dl_train))
images, labels = next(iter(dl_test))

model = torchvision.models.resnet18(pretrained=True)

model.fc = torch.nn.Linear(in_features=512, out_features=2)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)


def train(epochs):
    print('Starting training..')
    for e in range(0, epochs):
        
        print('='*20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('='*20)
        
        train_loss = 0
        val_loss = 0
        train_accuracy = 0
        device = "cpu"
        model.train().to(device)
        
        for train_step, (images, labels) in enumerate(dl_test):
            model.eval()
            optimizer.zero_grad()
            outputs= model(images.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_accuracy += sum((preds.to(device) == labels.to(device)).cpu().numpy())
            
            if train_step % 20 == 0:
                print("Evlauating...")
                accuracy = 0
                model.eval()
                for val_step, (images, labels) in enumerate(dl_test):
                    outputs = model(images.to(device))
                    loss = loss_fn(outputs, labels.to(device))
                    val_loss += loss.item()
                    
                    _, preds = torch.max(outputs, 1)
                    
                    accuracy += sum((preds.to(device) == labels.to(device)).cpu().numpy())
                val_loss /= (val_step + 1)
                accuracy = accuracy/len(test_dataset)
                print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

                if accuracy >= 0.98:
                    print('Performance condition satisfied, stopping..')
                    print("Saving the model")
                    torch.save(model, 'resnset.pt')
                    return

        train_loss /= (train_step + 1)

        print(f'Training Loss: {train_loss:.4f}')
    print('Training complete..')

train(3)
