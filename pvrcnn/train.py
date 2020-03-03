import os
import os.path as osp
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import multiprocessing

from pvrcnn.detector import ProposalLoss, PV_RCNN, Second
from pvrcnn.core import cfg, TrainPreprocessor, VisdomLinePlotter
from torch.utils.tensorboard import SummaryWriter
from pvrcnn.dataset import KittiDatasetTrain


def build_train_dataloader(cfg, preprocessor):
    dataloader = DataLoader(
        KittiDatasetTrain(cfg),
        collate_fn=preprocessor.collate,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=3,
    )
    return dataloader


def save_cpkt(model, optimizer, epoch, meta=None):
    fpath = f'./ckpts/epoch_{epoch}.pth'
    ckpt = dict(
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
        epoch=epoch,
        meta=meta,
    )
    os.makedirs('./ckpts', exist_ok=True)
    torch.save(ckpt, fpath)


def load_ckpt(fpath, model, optimizer):
    if not osp.isfile(fpath):
        print('{} not exist!'.format(fpath))
        return 0
    ckpt = torch.load(fpath)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    print('{} load done!'.format(fpath))
    epoch = ckpt['epoch']
    return epoch


def update_plot(losses, prefix):
    for key in ['loss', 'cls_loss', 'reg_loss']:
        plotter.update(f'{prefix}_{key}', losses[key].item())


def to_device(item):
    keys = ['G_cls', 'G_reg', 'M_cls', 'M_reg', 'points',
        'features', 'coordinates', 'occupancy']
    for key in keys:
        item[key] = item[key].cuda()


def train_model(model, dataloader, optimizer, lr_scheduler, loss_fn, epochs, start_epoch=0):
    model.train()
    for epoch in range(start_epoch, epochs):
        for step, item in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}')):
            to_device(item)
            optimizer.zero_grad()
            out = model(item)
            losses = loss_fn(out)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
            optimizer.step()
            print('epoch:{}, step:{}, loss:{}, cls_loss:{}, reg_loss:{}'.format(
                epoch, step, losses['loss'].item(), losses['cls_loss'].item(), losses['reg_loss'].item()))
            writer.add_scalar('Train_car/total_loss', losses['loss'].item(), epoch*dataloader.__len__()+step)
            writer.add_scalar('Train_car/cls_loss', losses['cls_loss'].item(), epoch*dataloader.__len__()+step)
            writer.add_scalar('Train_car/reg_loss', losses['reg_loss'].item(), epoch*dataloader.__len__()+step)
            writer.flush()
            lr_scheduler.step()
        save_cpkt(model, optimizer, epoch)


def get_proposal_parameters(model):
    for p in model.roi_grid_pool.parameters():
        p.requires_grad = False
    for p in model.refinement_layer.parameters():
        p.requires_grad = False
    return model.parameters()


def main():
    """TODO: Trainer class to manage objects."""
    model = Second(cfg).cuda()
    # model = PV_RCNN(cfg).cuda()
    parameters = model.parameters()
    loss_fn = ProposalLoss(cfg)
    preprocessor = TrainPreprocessor(cfg)
    dataloader = build_train_dataloader(cfg, preprocessor)
    parameters = get_proposal_parameters(model)
    optimizer = torch.optim.Adam(parameters, lr=cfg.TRAIN.LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
        steps_per_epoch=len(dataloader), epochs=cfg.TRAIN.EPOCHS)
    start_epoch = load_ckpt('./ckpts/epoch_20.pth', model, optimizer)
    train_model(model, dataloader, optimizer, scheduler, loss_fn, cfg.TRAIN.EPOCHS, start_epoch)


if __name__ == '__main__':
    writer = SummaryWriter(os.path.expanduser('~/log/'))
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    cfg.merge_from_file('../configs/second/car.yaml')
    main()
