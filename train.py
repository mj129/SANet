from net.sanet import SANet
import time
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.bbox_reader import BboxReader
from utils.util import Logger
from config import train_config, data_config, net_config, config
import pprint
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
import torch
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import random
import traceback
from torch.utils.tensorboard import SummaryWriter


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
this_module = sys.modules[__name__]

parser = argparse.ArgumentParser(description='PyTorch Detector')
parser.add_argument('--net', '-m', metavar='NET', default=train_config['net'],
                    help='neural net')
parser.add_argument('--epochs', default=train_config['epochs'], type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=train_config['batch_size'], type=int, metavar='N',
                    help='batch size')
parser.add_argument('--epoch-rcnn', default=train_config['epoch_rcnn'], type=int, metavar='NR',
                    help='number of epochs before training rcnn')
parser.add_argument('--ckpt', default=train_config['initial_checkpoint'], type=str, metavar='CKPT',
                    help='checkpoint to use')
parser.add_argument('--optimizer', default=train_config['optimizer'], type=str, metavar='SPLIT',
                    help='which split set to use')
parser.add_argument('--init-lr', default=train_config['init_lr'], type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=train_config['momentum'], type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=train_config['weight_decay'], type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--epoch-save', default=train_config['epoch_save'], type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--out-dir', default=train_config['out_dir'], type=str, metavar='OUT',
                    help='directory to save results of this training')
parser.add_argument('--train-set', default=train_config['train_set_list'], nargs='+', type=str,
                    help='train set paths list')
parser.add_argument('--val-set', default=train_config['val_set_list'], nargs='+', type=str,
                    help='val set paths list')
parser.add_argument('--data-dir', default=train_config['DATA_DIR'], type=str, metavar='OUT',
                    help='path to load data')
parser.add_argument('--num-workers', default=train_config['num_workers'], type=int, metavar='N',
                    help='number of data loading workers')


def main():
    # Load training configuration
    args = parser.parse_args()

    net = args.net
    initial_checkpoint = args.ckpt
    out_dir = args.out_dir
    weight_decay = args.weight_decay
    momentum = args.momentum
    optimizer = args.optimizer
    init_lr = args.init_lr
    epochs = args.epochs
    epoch_save = args.epoch_save
    epoch_rcnn = args.epoch_rcnn
    batch_size = args.batch_size
    lr_schdule = train_config['lr_schedule']
    data_dir = args.data_dir

    train_dataset = BboxReader(data_dir, args.train_set, config, mode='train')
    val_dataset = BboxReader(data_dir, args.val_set, config, mode='val')
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate)
    
    
    # Initilize network
    net = getattr(this_module, net)(net_config)
    net = net.cuda()
    
    optimizer = getattr(torch.optim, optimizer)
    optimizer = optimizer(net.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=momentum)

    start_epoch = 0

    if initial_checkpoint:
        print('[Loading model from %s]' % initial_checkpoint)
        checkpoint = torch.load(initial_checkpoint)
        start_epoch = checkpoint['epoch']

        state = net.state_dict()
        state.update(checkpoint['state_dict'])

        try:
            net.load_state_dict(state)
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print('Load something failed!')
            traceback.print_exc()

    start_epoch = start_epoch + 1

    model_out_dir = os.path.join(out_dir, 'model')
    tb_out_dir = os.path.join(out_dir, 'runs')
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    logfile = os.path.join(out_dir, 'log_train')
    sys.stdout = Logger(logfile)

    print('[Training configuration]')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    print('[Model configuration]')
    pprint.pprint(net_config)

    print('[start_epoch %d, out_dir %s]' % (start_epoch, out_dir))
    print('[length of train loader %d, length of valid loader %d]' % (len(train_loader), len(val_loader)))

    # Write graph to tensorboard for visualization
    writer = SummaryWriter(tb_out_dir)
    train_writer = SummaryWriter(os.path.join(tb_out_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(tb_out_dir, 'val'))
    # writer.add_graph(net, (torch.zeros((16, 1, 128, 128, 128)).cuda(), [[]], [[]], [[]], [torch.zeros((16, 128, 128, 128))]), verbose=False)

    for i in tqdm(range(start_epoch, epochs + 1), desc='Total'):
        # learning rate schedule
        if isinstance(optimizer, torch.optim.SGD):
            lr = lr_schdule(i, init_lr=init_lr, total=epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = np.inf

        if i >= epoch_rcnn:
            net.use_rcnn = True
        else:
            net.use_rcnn = False

        print('[epoch %d, lr %f, use_rcnn: %r]' % (i, lr, net.use_rcnn))
        train(net, train_loader, optimizer, i, train_writer)
        validate(net, val_loader, i, val_writer)


        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        if i % epoch_save == 0:
            torch.save({
                'epoch': i,
                'out_dir': out_dir,
                'state_dict': state_dict,
                'optimizer' : optimizer.state_dict()},
                os.path.join(model_out_dir, '%03d.ckpt' % i))

    writer.close()
    train_writer.close()
    val_writer.close()


def train(net, train_loader, optimizer, epoch, writer):
    net.set_mode('train')
    s = time.time()
    rpn_cls_loss, rpn_reg_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []
    total_loss = []
    rpn_stats = []
    rcnn_stats = []

    for j, (input, truth_box, truth_label) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train %d' % epoch):
        input = Variable(input).cuda()
        truth_box = np.array(truth_box)
        truth_label = np.array(truth_label)

        net(input, truth_box, truth_label)

        loss, rpn_stat, rcnn_stat = net.loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rpn_cls_loss.append(net.rpn_cls_loss.cpu().data.item())
        rpn_reg_loss.append(net.rpn_reg_loss.cpu().data.item())
        rcnn_cls_loss.append(net.rcnn_cls_loss.cpu().data.item())
        rcnn_reg_loss.append(net.rcnn_reg_loss.cpu().data.item())

        total_loss.append(loss.cpu().data.item())
        rpn_stats.append(rpn_stat)
        rcnn_stats.append(rcnn_stat)

        del input, truth_box, truth_label
        del net.rpn_proposals, net.detections
        del net.total_loss, net.rpn_cls_loss, net.rpn_reg_loss, net.rcnn_cls_loss, net.rcnn_reg_loss
        del net.rpn_logits_flat, net.rpn_deltas_flat

        if net.use_rcnn:
            del net.rcnn_logits, net.rcnn_deltas

        torch.cuda.empty_cache()

    rpn_stats = np.asarray(rpn_stats, np.float32)
    
    print('Train Epoch %d, iter %d, total time %f, loss %f' % (epoch, j, time.time() - s, np.average(total_loss)))
    print('rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f' % \
        (np.average(rpn_cls_loss), np.average(rpn_reg_loss),
            np.average(rcnn_cls_loss), np.average(rcnn_reg_loss)
            ))
    print('rpn_stats: tpr %f, tnr %f, total pos %d, total neg %d, reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
        100.0 * np.sum(rpn_stats[:, 0]) / np.sum(rpn_stats[:, 1]),
        100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3]),
        np.sum(rpn_stats[:, 1]),
        np.sum(rpn_stats[:, 3]),
        np.mean(rpn_stats[:, 4]),
        np.mean(rpn_stats[:, 5]),
        np.mean(rpn_stats[:, 6]),
        np.mean(rpn_stats[:, 7]),
        np.mean(rpn_stats[:, 8]),
        np.mean(rpn_stats[:, 9])))

    # Write to tensorboard
    writer.add_scalar('loss', np.average(total_loss), epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
    writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
    writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)

    writer.add_scalar('rpn_reg_z', np.mean(rpn_stats[:, 4]), epoch)
    writer.add_scalar('rpn_reg_y', np.mean(rpn_stats[:, 5]), epoch)
    writer.add_scalar('rpn_reg_x', np.mean(rpn_stats[:, 6]), epoch)
    writer.add_scalar('rpn_reg_d', np.mean(rpn_stats[:, 7]), epoch)
    writer.add_scalar('rpn_reg_h', np.mean(rpn_stats[:, 8]), epoch)
    writer.add_scalar('rpn_reg_w', np.mean(rpn_stats[:, 9]), epoch)

    if net.use_rcnn:
        confusion_matrix = np.asarray([stat[-1] for stat in rcnn_stats], np.int32)
        rcnn_stats = np.asarray([stat[:-1] for stat in rcnn_stats], np.float32)
        
        confusion_matrix = np.sum(confusion_matrix, 0)

        print('rcnn_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(rcnn_stats[:, 0]),
            np.mean(rcnn_stats[:, 1]),
            np.mean(rcnn_stats[:, 2]),
            np.mean(rcnn_stats[:, 3]),
            np.mean(rcnn_stats[:, 4]),
            np.mean(rcnn_stats[:, 5])))
        # print_confusion_matrix(confusion_matrix)
        writer.add_scalar('rcnn_reg_z', np.mean(rcnn_stats[:, 0]), epoch)
        writer.add_scalar('rcnn_reg_y', np.mean(rcnn_stats[:, 1]), epoch)
        writer.add_scalar('rcnn_reg_x', np.mean(rcnn_stats[:, 2]), epoch)
        writer.add_scalar('rcnn_reg_d', np.mean(rcnn_stats[:, 3]), epoch)
        writer.add_scalar('rcnn_reg_h', np.mean(rcnn_stats[:, 4]), epoch)
        writer.add_scalar('rcnn_reg_w', np.mean(rcnn_stats[:, 5]), epoch)
    

def validate(net, val_loader, epoch, writer):
    net.set_mode('valid')
    rpn_cls_loss, rpn_reg_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []
    total_loss = []
    rpn_stats = []
    rcnn_stats = []

    s = time.time()
    for j, (input, truth_box, truth_label) in tqdm(enumerate(val_loader), total=len(val_loader), desc='Val %d' % epoch):
        with torch.no_grad():
            input = Variable(input).cuda()
            truth_box = np.array(truth_box)
            truth_label = np.array(truth_label)

            net(input, truth_box, truth_label)
            loss, rpn_stat, rcnn_stat = net.loss()

        rpn_cls_loss.append(net.rpn_cls_loss.cpu().data.item())
        rpn_reg_loss.append(net.rpn_reg_loss.cpu().data.item())
        rcnn_cls_loss.append(net.rcnn_cls_loss.cpu().data.item())
        rcnn_reg_loss.append(net.rcnn_reg_loss.cpu().data.item())

        total_loss.append(loss.cpu().data.item())
        rpn_stats.append(rpn_stat)
        rcnn_stats.append(rcnn_stat)

    rpn_stats = np.asarray(rpn_stats, np.float32)
    print('Val Epoch %d, iter %d, total time %f, loss %f' % (epoch, j, time.time()-s, np.average(total_loss)))
    print('rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f' % \
        (np.average(rpn_cls_loss), np.average(rpn_reg_loss),
            np.average(rcnn_cls_loss), np.average(rcnn_reg_loss)
            ))
    print('rpn_stats: tpr %f, tnr %f, total pos %d, total neg %d, reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
        100.0 * np.sum(rpn_stats[:, 0]) / np.sum(rpn_stats[:, 1]),
        100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3]),
        np.sum(rpn_stats[:, 1]),
        np.sum(rpn_stats[:, 3]),
        np.mean(rpn_stats[:, 4]),
        np.mean(rpn_stats[:, 5]),
        np.mean(rpn_stats[:, 6]),
        np.mean(rpn_stats[:, 7]),
        np.mean(rpn_stats[:, 8]),
        np.mean(rpn_stats[:, 9])))
    
    # Write to tensorboard
    writer.add_scalar('loss', np.average(total_loss), epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
    writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
    writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)

    writer.add_scalar('rpn_reg_z', np.mean(rpn_stats[:, 4]), epoch)
    writer.add_scalar('rpn_reg_y', np.mean(rpn_stats[:, 5]), epoch)
    writer.add_scalar('rpn_reg_x', np.mean(rpn_stats[:, 6]), epoch)
    writer.add_scalar('rpn_reg_d', np.mean(rpn_stats[:, 7]), epoch)
    writer.add_scalar('rpn_reg_h', np.mean(rpn_stats[:, 8]), epoch)
    writer.add_scalar('rpn_reg_w', np.mean(rpn_stats[:, 9]), epoch)

    if net.use_rcnn:
        confusion_matrix = np.asarray([stat[-1] for stat in rcnn_stats], np.int32)
        rcnn_stats = np.asarray([stat[:-1] for stat in rcnn_stats], np.float32)
        
        confusion_matrix = np.sum(confusion_matrix, 0)
        print('rcnn_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(rcnn_stats[:, 0]),
            np.mean(rcnn_stats[:, 1]),
            np.mean(rcnn_stats[:, 2]),
            np.mean(rcnn_stats[:, 3]),
            np.mean(rcnn_stats[:, 4]),
            np.mean(rcnn_stats[:, 5])))
        # print_confusion_matrix(confusion_matrix)
        writer.add_scalar('rcnn_reg_z', np.mean(rcnn_stats[:, 0]), epoch)
        writer.add_scalar('rcnn_reg_y', np.mean(rcnn_stats[:, 1]), epoch)
        writer.add_scalar('rcnn_reg_x', np.mean(rcnn_stats[:, 2]), epoch)
        writer.add_scalar('rcnn_reg_d', np.mean(rcnn_stats[:, 3]), epoch)
        writer.add_scalar('rcnn_reg_h', np.mean(rcnn_stats[:, 4]), epoch)
        writer.add_scalar('rcnn_reg_w', np.mean(rcnn_stats[:, 5]), epoch)
    
    del input, truth_box, truth_label
    del net.rpn_proposals, net.detections
    del net.total_loss, net.rpn_cls_loss, net.rpn_reg_loss, net.rcnn_cls_loss, net.rcnn_reg_loss

    if net.use_rcnn:
        del net.rcnn_logits, net.rcnn_deltas

    torch.cuda.empty_cache()

def print_confusion_matrix(confusion_matrix):
    line_new = '{:>4}  ' * (len(config['roi_names']) + 2)
    print(line_new.format('gt/p', *list(range(len(config['roi_names']) + 1))))

    for i in range(len(config['roi_names']) + 1):
        print(line_new.format(i, *list(confusion_matrix[i])))
        

if __name__ == '__main__':
    main()



