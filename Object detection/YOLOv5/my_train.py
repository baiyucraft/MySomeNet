import math
import os
import random
from copy import deepcopy

import numpy as np
import torch
import yaml
from pathlib import Path
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm

import val
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import init_seeds, one_cycle, check_img_size, colorstr, labels_to_class_weights
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_labels
from utils.torch_utils import select_device, intersect_dicts, ModelEMA, EarlyStopping


class Config:
    root = 'my_config'
    hyp = 'pest_hyp.yaml'
    device = select_device()
    # data
    imgsz = 640
    data_path = 'pest_data.yaml'
    val = False
    autoanchor = True
    cache = None  # ram or disk

    # model
    cfg = 'pest_s.yaml'
    pretrained = 'model_data/pest_s/last.pt'
    num_workers = 2
    resume = True

    # train param
    patience = 100
    batch_size = 3
    epochs = 300
    multi_scale = True
    # loss
    label_smoothing = 0.0


if __name__ == '__main__':
    # create folder
    init_seeds()
    root = Path(Config.root)
    model_name = Config.cfg[:-5]
    save_dir = Path('model_data') / model_name
    save_dir.mkdir(exist_ok=True)
    last, best = save_dir / 'last.pt', save_dir / 'best.pt'

    # load hyp
    with open(root / Config.hyp, 'r', encoding='utf-8') as f:
        hyp = yaml.safe_load(f)
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)

    # load data_dic
    data_path = root / Config.data_path
    with open(data_path, 'r', encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)
    nc, names = data_dict['nc'], data_dict['names']
    # train_path, val_path = data_path / data_dict['train'], data_path / data_dict['val']
    train_path, val_path = data_dict['train'], data_dict['val']
    # train_path = './datasets/pest_test/images/train'

    # load cfg
    with open(root / Config.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # model
    model = Model(cfg, ch=3, nc=nc)

    # optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / Config.batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= Config.batch_size * accumulate / nbs  # scale weight_decay
    # optimizer parameter
    g0, g1, g2 = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)
    optimizer = torch.optim.SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': g2})
    del g0, g1, g2

    # scheduler
    lf = one_cycle(1, hyp['lrf'], Config.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model)

    model.to(Config.device)
    # Resume
    start_epoch, best_fitness = 0, 0.0
    if Config.pretrained and Config.resume:
        # model
        ckpt = torch.load(Config.pretrained, map_location=Config.device)
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
        model.load_state_dict(csd, strict=False)
        # optimizer
        if ckpt['optimizer']:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
        # epochs
        start_epoch = ckpt['epoch'] + 1

        del ckpt, csd

    # image sizes
    gs = max(int(model.stride.max()), 32)
    nl = model.model[-1].nl
    imgsz = check_img_size(Config.imgsz, gs, floor=gs * 2)

    # train loader
    train_loader, dataset = create_dataloader(train_path, imgsz, Config.batch_size, gs, hyp=hyp, augment=True,
                                              cache=Config.cache, workers=Config.num_workers,
                                              prefix=colorstr('train: '))
    # val loader
    val_loader, _ = create_dataloader(val_path, imgsz, Config.batch_size, gs, hyp=hyp,
                                      cache=Config.cache if Config.val else None, rect=True,
                                      workers=Config.num_workers, prefix=colorstr('val: '))
    # max label class
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())
    # number of batches
    nb = len(train_loader)
    # labels anchors
    if not Config.resume:
        labels = np.concatenate(dataset.labels, 0)
        plot_labels(labels, names, save_dir)

        # Anchors
        if Config.autoanchor:
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
        model.half().float()  # pre-reduce anchor precision

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = Config.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(Config.device) * nc
    model.names = names

    # init train
    last_opt_step = -1
    # number of warmup iterations
    # nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    # mAP per class
    maps = np.zeros(nc)
    # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1
    stopper = EarlyStopping(patience=Config.patience)
    compute_loss = ComputeLoss(model)

    # train
    for epoch in range(start_epoch, Config.epochs):
        model.train()
        mloss = torch.zeros(3, device=Config.device)
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=nb)
        optimizer.zero_grad()

        # one batch
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(Config.device, non_blocking=True).float() / 255.0

            if Config.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(Config.device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # loss log
            mloss = (mloss * i + loss_items) / (i + 1)
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                f'{epoch}/{Config.epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
        # end batch

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        final_epoch = (epoch + 1 == Config.epochs) or stopper.possible_stop
        if Config.val or final_epoch:  # Calculate mAP
            results, maps, _ = val.run(data_dict, batch_size=Config.batch_size * 2, imgsz=imgsz, model=ema.ema,
                                       dataloader=val_loader, save_dir=save_dir, plots=False, compute_loss=compute_loss)
        # update
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi
        log_vals = list(mloss) + list(results) + lr

        # save
        ckpt = {'epoch': epoch,
                'best_fitness': best_fitness,
                'model': deepcopy(model).half(),
                'ema': deepcopy(ema.ema).half(),
                'updates': ema.updates,
                'optimizer': optimizer.state_dict()}
        torch.save(ckpt, last)
        if best_fitness == fi:
            torch.save(ckpt, best)
        del ckpt
