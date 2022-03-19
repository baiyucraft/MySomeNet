import os

import torch
from torch.utils import data

import network
import utils
from datasets.my_data import MyDataSet

config = {
    'model': 'deeplabv3plus_resnet50',
    'dataset': 'my',
    'ckpt': 'checkpoints/latest_deeplabv3plus_resnet50_my_os8.pth',
    'total_epochs': 100,
    'batch_size': 8,
    'lr': 0.01,
    'weight_decay': 1e-4,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'scheduler_param': {
        'T_0': 100,
        'T_mult': 100,
        'eta_min': 1e-5
    }
}


def save_ckpt(path):
    """ save current model
    """
    torch.save({
        "cur_epochs": cur_epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
    }, path)
    print(f"Model saved as {path} by {best_score:.3f}")


if __name__ == '__main__':
    train_data = MyDataSet()
    train_loader = data.DataLoader(train_data, config['batch_size'], shuffle=True, num_workers=2, drop_last=True)
    model = network.modeling.deeplabv3plus_resnet50(num_classes=2)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    # Set up
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * config['lr']},
        {'params': model.classifier.parameters(), 'lr': config['lr']},
    ], lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **config['scheduler_param'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    # Set up criterion
    criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    utils.mkdir('checkpoints')

    # Restore
    best_score = 0.0
    cur_epochs = 0
    if config['ckpt'] is not None and os.path.isfile(config['ckpt']):
        checkpoint = torch.load(config['ckpt'], map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model.to(config['device'])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epochs = checkpoint["cur_epochs"]
        best_score = checkpoint['best_score']
        print("Training state restored from %s" % config['ckpt'])
    else:
        print("[!] Retrain")
        model.to(config['device'])

    interval_loss = 0.0
    while cur_epochs <= config['total_epochs']:
        model.train()
        cur_epochs += 1
        epoch_loss = []
        for (images, labels) in train_loader:
            images, labels = images.to(config['device']), labels.to(config['device']).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().item()
            epoch_loss.append(np_loss)

        per_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        interval_loss += per_epoch_loss
        if per_epoch_loss <= best_score:
            best_score = per_epoch_loss
            save_ckpt(config['ckpt'])
        if cur_epochs % 5 == 0:
            interval_loss = interval_loss / 5
            print(f"Epoch {cur_epochs}/{config['total_epochs']}, Loss={interval_loss :.3f}")
            interval_loss = 0.0

        scheduler.step()
