
from torch_utils.engine import train_one_epoch, evaluate
from datasets import CustomDataset, get_train_transform, DataLoader, get_valid_transform
from utils.general import set_training_dir, Averager,show_tranformed_image, SaveBestModel
import torch
import yaml
import numpy as np
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as transforms



torch.multiprocessing.set_sharing_strategy('file_system')

np.random.seed(42)


def get_train_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })


def collate_fn(batch):
    return tuple(zip(*batch))

def create_train_dataset(
    train_dir_images, train_dir_labels, 
    resize_width, resize_height, classes,
    use_train_aug=False,
    mosaic=True
):
    train_dataset = CustomDataset(
        train_dir_images, train_dir_labels,
        resize_width, resize_height, classes, 
        get_train_transform(),
        use_train_aug=use_train_aug,
        train=True, mosaic=mosaic
    )
    return train_dataset

def create_valid_dataset(
    valid_dir_images, valid_dir_labels, 
    resize_width, resize_height, classes
):
    valid_dataset = CustomDataset(
        valid_dir_images, valid_dir_labels, 
        resize_width, resize_height, classes, 
        get_valid_transform(),
        train=False
    )
    return valid_dataset

def create_train_loader(train_dataset, batch_size, num_workers=1):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader
def create_valid_loader(valid_dataset, batch_size, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader


def main():
    with open("dataset_config.yaml") as file:
        data_configs = yaml.safe_load(file)
    
    TRAIN_DIR_IMAGES = data_configs['TRAIN_DIR_IMAGES']
    TRAIN_DIR_LABELS = data_configs['TRAIN_DIR_LABELS']
    VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
    VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    CLASSES = data_configs['CLASSES']
    NUM_WORKERS = 2
    DEVICE = "cpu"
    SAVE_VALID_PREDICTIONS = data_configs['SAVE_VALID_PREDICTION_IMAGES']
    BATCH_SIZE = 20
    VISUALIZE_TRANSFORMED_IMAGES = False
    OUT_DIR = set_training_dir("results")
    COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))

    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 640
    
    train_dataset = create_train_dataset(
        TRAIN_DIR_IMAGES, TRAIN_DIR_LABELS,
        IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES,
        use_train_aug=False,
        mosaic=False
    )
    valid_dataset = create_valid_dataset(
        VALID_DIR_IMAGES, VALID_DIR_LABELS, 
        IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES
    )

    train_loader = create_train_loader(train_dataset, BATCH_SIZE, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    train_loss_hist = Averager()
    train_loss_list = []
    loss_cls_list = []
    loss_box_reg_list = []
    loss_objectness_list = []
    loss_rpn_list = []
    train_loss_list_epoch = []
    val_map_05 = []
    val_map = []
    start_epochs = 0

    print('Building model from scratch...')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )

    model = model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
    save_best_model = SaveBestModel()

    for epoch in range(start_epochs, 2):

        _, batch_loss_list, \
             batch_loss_cls_list, \
             batch_loss_box_reg_list, \
             batch_loss_objectness_list, \
             batch_loss_rpn_list = train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            DEVICE, 
            epoch, 
            train_loss_hist,
            print_freq=100,
            scheduler=None
        )

        _, stats, _ = evaluate(
            model, 
            valid_loader, 
            device=DEVICE,
            save_valid_preds=SAVE_VALID_PREDICTIONS,
            out_dir=OUT_DIR,
            classes=CLASSES,
            colors=COLORS
        )

        train_loss_list.extend(batch_loss_list)
        loss_cls_list.extend(batch_loss_cls_list)
        loss_box_reg_list.extend(batch_loss_box_reg_list)
        loss_objectness_list.extend(batch_loss_objectness_list)
        loss_rpn_list.extend(batch_loss_rpn_list)
        train_loss_list_epoch.append(train_loss_hist.value)
        val_map_05.append(stats[1])
        val_map.append(stats[0])
        torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss_list': train_loss_list,
                    'train_loss_list_epoch': train_loss_list_epoch,
                    'val_map': val_map,
                    'val_map_05': val_map_05,
                    'config': data_configs,
                    'model_name': "fasterrcnn_resnet50_fpn_v2"
                    }, f"{OUT_DIR}/faster_cnn.pth")
        
        save_best_model(
            model, 
            val_map[-1], 
            epoch, 
            OUT_DIR,
            data_configs,
            "fasterrcnn_resnet50_fpn_v2"
        )

if __name__ == '__main__':
    main()