import wandb

from Tools.utils import try_gpu, load_caltech_256
from net import MobileNetV3

import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.notebook import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2 ** 32 - 1)
np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

device = try_gpu()

config = dict(
    epochs=50,
    classes=257,
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=5e-4)


def model_pipeline(hyperparameters):
    # tell wandb to get started
    with wandb.init(project="test", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer, optimizer_adjust = make(config)
        # print(model)

        # and use them to train the model
        train(model, train_loader, criterion, optimizer, optimizer_adjust, config)

        # and test its final performance
        test(model, test_loader)

    return model


def make(config):
    # Make the data
    train_iter, test_iter = load_caltech_256(16)

    # Make the model
    model = MobileNetV3(257).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
                                momentum=0.9)
    optimizer_adjust = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=1e-5)

    return model, train_iter, test_iter, criterion, optimizer, optimizer_adjust


# train
def train(model, loader, criterion, optimizer, optimizer_adjust, config):
    # tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct += len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)
        optimizer_adjust.step()


def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)

    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def test(model, test_loader):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {100 * correct / total}%")

        wandb.log({"test_accuracy": correct / total})

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")


if __name__ == '__main__':
    model = model_pipeline(config)
