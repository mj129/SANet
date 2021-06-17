import numpy as np
import torch
import os
import traceback
import time
import nrrd
import sys
import matplotlib.pyplot as plt
import logging
import argparse
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel
from scipy.ndimage.measurements import label
from scipy.ndimage import center_of_mass
from net.sanet import SANet
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.bbox_reader import BboxReader
from config import config
import pandas as pd
from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation

this_module = sys.modules[__name__]
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--net', '-m', metavar='NET', default=config['net'],
                    help='neural net')
parser.add_argument("--mode", type=str, default = 'eval',
                    help="you want to test or val")
parser.add_argument("--weight", type=str, default='./results/model/model.ckpt',
                    help="path to model weights to be used")
parser.add_argument("--dicom-path", type=str, default=None,
                    help="path to dicom files of patient")
parser.add_argument("--out-dir", type=str, default=config['out_dir'],
                    help="path to save the results")
parser.add_argument("--test-set-name", type=str, default=config['test_set_name'],
                    help="path to save the results")


def main():
    logging.basicConfig(format='[%(levelname)s][%(asctime)s] %(message)s', level=logging.INFO)
    args = parser.parse_args()

    if args.mode == 'eval':
        data_dir = config['preprocessed_data_dir']
        test_set_name = args.test_set_name
        num_workers = 16
        initial_checkpoint = args.weight
        net = args.net
        out_dir = args.out_dir

        net = getattr(this_module, net)(config)
        net = net.cuda()

        if initial_checkpoint:
            print('[Loading model from %s]' % initial_checkpoint)
            checkpoint = torch.load(initial_checkpoint)
            epoch = checkpoint['epoch']

            net.load_state_dict(checkpoint['state_dict'])
        else:
            print('No model weight file specified')
            return

        print('out_dir', out_dir)
        save_dir = os.path.join(out_dir, 'res', str(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'FROC')):
            os.makedirs(os.path.join(save_dir, 'FROC'))

        dataset = BboxReader(data_dir, test_set_name, config, mode='eval')
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                 num_workers=num_workers, pin_memory=False, collate_fn=train_collate)
        eval(net, test_loader, save_dir)
    else:
        logging.error('Mode %s is not supported' % (args.mode))


def eval(net, dataset, save_dir=None):
    net.set_mode('eval')
    net.use_rcnn = True

    print('Total # of eval data %d' % (len(dataset)))
    for i, (input, truth_bboxes, truth_labels) in enumerate(dataset):
        try:
            input = Variable(input).cuda()
            truth_bboxes = np.array(truth_bboxes)
            truth_labels = np.array(truth_labels)
            pid = dataset.dataset.filenames[i]

            print('[%d] Predicting %s' % (i, pid))

            with torch.no_grad():
                net.forward(input, truth_bboxes, truth_labels)

            detections = net.rpn_proposals.cpu().numpy()

            print('detections', detections.shape)

            if len(detections):
                detections = detections[:, 1:-1]
                np.save(os.path.join(save_dir, '%s_detections.npy' % (pid)), detections)

            # Clear gpu memory
            del input, truth_bboxes, truth_labels
            torch.cuda.empty_cache()

        except Exception as e:
            del input, truth_bboxes, truth_labels
            torch.cuda.empty_cache()
            traceback.print_exc()

            return
    
    # Generate prediction csv for the use of performning FROC analysis
    res = []
    for pid in dataset.dataset.filenames:
        if os.path.exists(os.path.join(save_dir, '%s_detections.npy' % (pid))):
            detections = np.load(os.path.join(save_dir, '%s_detections.npy' % (pid)))
            detections = detections[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(detections))
            res.append(np.concatenate([names, detections], axis=1))
    
    res = np.concatenate(res, axis=0)
    col_names = ['pid','center_x','center_y','center_z','diameter', 'probability']
    eval_dir = os.path.join(save_dir, 'FROC')
    res_path = os.path.join(eval_dir, 'results.csv')
    
    df = pd.DataFrame(res, columns=col_names)
    df.to_csv(res_path, index=False)

    # Start evaluating
    if not os.path.exists(os.path.join(eval_dir, 'res')):
        os.makedirs(os.path.join(eval_dir, 'res'))

    annotations_filename = '/home/media/ssd/process_zoom/split_full_with_nodule_9classes/test_anno_center.csv'
    # test_anno_center.csv is a csv file with 'center_x','center_y','center_z', and 'diameter'. You can obtain these parameters by using 'xmin', 'xmax', etc.
    val_path = '/home/media/ssd/process_zoom/split_full_with_nodule_9classes/test.txt'

    noduleCADEvaluation(annotations_filename, res_path, val_path, os.path.join(eval_dir, 'res'))

if __name__ == '__main__':
    main()
