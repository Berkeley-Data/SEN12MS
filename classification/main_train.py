# Modified from Jian Kang, https://www.rsim.tu-berlin.de/menue/team/dring_jian_kang/
# Modified by Yu-Lun Wu, TUM

import os
import argparse
import numpy as np
from datetime import datetime 
from tqdm import tqdm
import json

import torch
import torch.optim as optim 
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter

import shutil 
import sys
sys.path.append('../')

from dataset import SEN12MS, BigEarthNet, ToTensor, Normalize
from models.VGG import VGG16, VGG19
from models.ResNet import ResNet50, ResNet50_1x1, ResNet101, ResNet152, Moco, Moco_1x1, Moco_1x1RND
from models.DenseNet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from metrics import MetricTracker, Precision_score, Recall_score, F1_score, \
    F2_score, Hamming_loss, Subset_accuracy, Accuracy_score, One_error, \
    Coverage_error, Ranking_loss, LabelAvgPrec_score, calssification_report, \
    conf_mat_nor, get_AA, multi_conf_mat, OA_multi

import wandb

#sec.2 (done)
    
model_choices = ['VGG16', 'VGG19',
                 'Supervised','ResNet101','ResNet152', 'Supervised_1x1',
                 'DenseNet121','DenseNet161','DenseNet169','DenseNet201', 'Moco', 'Moco_1x1', 'Moco_1x1RND']
label_choices = ['multi_label', 'single_label']
sensor_choices = ['s1', 's2', 's1s2']

# ----------------------- define and parse arguments --------------------------
parser = argparse.ArgumentParser()

# experiment name
parser.add_argument('--exp_name', type=str, default=None,
                    help="experiment name. will be used in the path names \
                         for log- and savefiles. If no input experiment name, \
                         path would be set to model name.")

# data directory
parser.add_argument('--dataset', type=str, default=None,
                    help='dataset name. dataset should be at data/{dataset}/data ')
parser.add_argument('--label_split_dir', type=str, default=None,
                    help="path to label data and split list")
parser.add_argument('--data_size', type=str, default="full",
                    help="64, 128, 256, 1000, 1024, full")
# input/output
parser.add_argument('--use_fusion', action='store_true', default=False,
                    help='use 12 channels with zero padding')
parser.add_argument('--sensor_type', type=str, choices = sensor_choices,
                    default='s1s2',
                    help="s1, s2, or s1s2 (default: s1s2)")
# parser.add_argument('--use_s2', action='store_true', default=False,
#                     help='use sentinel-2 bands')
# parser.add_argument('--use_s1', action='store_true', default=False,
#                     help='use sentinel-1 data')
parser.add_argument('--use_RGB', action='store_true', default=False,
                    help='use sentinel-2 RGB bands')
parser.add_argument('--simple_scheme', action='store_true', default=False,
                    help='use IGBP simplified scheme; otherwise: IGBP original scheme')
parser.add_argument('--label_type', type=str, choices = label_choices,
                    default='multi_label',
                    help="label-type (default: multi_label)")
parser.add_argument('--threshold', type=float, default=0.1, 
                    help='threshold to convert probability-labels to multi-hot \
                    labels, mean/std for normalizatin would not be accurate \
                    if the threshold is larger than 0.22. \
                    for single_label threshold would be ignored')
parser.add_argument('--eval', action='store_true', default=False,
                    help='evaluate against test set')

# network
parser.add_argument('--model', type=str, choices = model_choices,
                    default='ResNet50',
                    help="network architecture (default: ResNet50)")

# training hyperparameters
parser.add_argument('--lr', type=float, default=0.001, 
                    help='initial learning rate')
parser.add_argument('--use_lr_step', action='store_true', default=False,
                    help='use learning rate steps')
parser.add_argument('--lr_step_size', type=int, default=25,
                    help='Learning rate step size')
parser.add_argument('--lr_step_gamma', type=float, default=0.1,
                    help='Learning rate step gamma')
parser.add_argument('--decay', type=float, default=1e-5,
                    help='decay rate')
parser.add_argument('--batch_size', type=int, default=64,
                    help='mini-batch size (default: 64)')
parser.add_argument('--num_workers',type=int, default=4,
                    help='num_workers for data loading in pytorch')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of training epochs (default: 100)')
parser.add_argument('--resume', '-r', type=str, default=None,
                    help='path to the pretrained weights file', )
parser.add_argument('--pt_dir', '-pd', type=str, default=None,
                    help='directory for pretrained model', )
parser.add_argument('--pt_name', '-pn', type=str, default=None,
                    help='model name without extension', )
parser.add_argument('--pt_type', '-pt', type=str, default='bb',
                    help='bb (backbone) or qe (query encoder)', )

# Dump predicted data
parser.add_argument('--output_pred', action='store_true', default=False,
                    help='Prediction data')
args = parser.parse_args()

wandb.init(config=args)

# -------------------- set directory for saving files -------------------------

if wandb.run is not None:
    # save to wandb run dir for tracking and saving the models
    checkpoint_dir = wandb.run.dir
    logs_dir = wandb.run.dir
elif args.exp_name:
    checkpoint_dir = os.path.join('./', args.exp_name, 'checkpoints')
    logs_dir = os.path.join('./', args.exp_name, 'logs')
else:
    checkpoint_dir = os.path.join('./', args.model, 'checkpoints')
    logs_dir = os.path.join('./', args.model, 'logs')

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.isdir(logs_dir):
    os.makedirs(logs_dir)

# ----------------------------- saving files ---------------------------------
sv_name_eval = '' # Used to save a file during the test evaluation

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, str(value)))

def save_checkpoint(state, is_best, name):

    filename = os.path.join(checkpoint_dir, name + '_checkpoint.pth')

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, name + 
                                               '_model_best.pth'))
        
# -------------------------------- Main Program -------------------------------
def main():
    global args
    global sv_name_eval
    # save configuration to file
    sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    sv_name_eval = sv_name
    print('saving file name is ', sv_name)

    write_arguments_to_file(args, os.path.join(logs_dir, sv_name+'_arguments.txt'))

# ----------------------------------- data
    # define mean/std of the training set (for data normalization)
    label_type = args.label_type
    use_s1 = (args.sensor_type == 's1') | (args.sensor_type == 's1s2')
    use_s2 = (args.sensor_type == 's2') | (args.sensor_type == 's1s2')

    dataset = args.dataset
    data_dir = os.path.join("data", dataset, "data")

    bands_mean = {}
    bands_std = {}
    train_dataGen = None
    val_dataGen = None
    test_dataGen = None

    print(f"Using {dataset} dataset")
    if dataset == 'sen12ms':
        bands_mean = {'s1_mean': [-11.76858, -18.294598],
                      's2_mean': [1226.4215, 1137.3799, 1139.6792, 1350.9973, 1932.9058,
                                  2211.1584, 2154.9846, 2409.1128, 2001.8622, 1356.0801]}
        bands_std = {'s1_std': [4.525339, 4.3586307],
                     's2_std': [741.6254, 740.883, 960.1045, 946.76056, 985.52747,
                                1082.4341, 1057.7628, 1136.1942, 1132.7898, 991.48016]}
    elif dataset == 'bigearthnet':
        # THE S2 BAND STATISTICS WERE PROVIDED BY THE BIGEARTHNET TEAM
        # Source: https://git.tu-berlin.de/rsim/bigearthnet-models-tf/-/blob/master/BigEarthNet.py
        bands_mean = {'s1_mean': [-12.619993, -19.290445],
                      's2_mean': [340.76769064,429.9430203,614.21682446,590.23569706,950.68368468,1792.46290469,
                                  2075.46795189,2218.94553375,2266.46036911,2246.0605464,1594.42694882,1009.32729131]}
        bands_std = {'s1_std': [5.115911, 5.464428],
                     's2_std': [554.81258967,572.41639287,582.87945694,675.88746967,729.89827633,1096.01480586,
                                1273.45393088,1365.45589904,1356.13789355,1302.3292881,1079.19066363,818.86747235]}
    else:
        raise NameError(f"unknown dataset: {dataset}")

    # load datasets 
    imgTransform = transforms.Compose([ToTensor(),Normalize(bands_mean, bands_std)])
    if dataset == 'sen12ms':
        train_dataGen = SEN12MS(data_dir, args.label_split_dir,
                                imgTransform=imgTransform,
                                label_type=label_type, threshold=args.threshold, subset="train",
                                use_s1=use_s1, use_s2=use_s2, use_RGB=args.use_RGB,
                                IGBP_s=args.simple_scheme, data_size=args.data_size, sensor_type=args.sensor_type, use_fusion=args.use_fusion)

        val_dataGen = SEN12MS(data_dir, args.label_split_dir,
                              imgTransform=imgTransform,
                              label_type=label_type, threshold=args.threshold, subset="val",
                              use_s1=use_s1, use_s2=use_s2, use_RGB=args.use_RGB,
                              IGBP_s=args.simple_scheme, data_size=args.data_size, sensor_type=args.sensor_type, use_fusion=args.use_fusion)

        if args.eval:
            test_dataGen = SEN12MS(data_dir, args.label_split_dir,
                                   imgTransform=imgTransform,
                                   label_type=label_type, threshold=args.threshold, subset="test",
                                   use_s1=use_s1, use_s2=use_s2, use_RGB=args.use_RGB,
                                   IGBP_s=args.simple_scheme, sensor_type=args.sensor_type, use_fusion=args.use_fusion)
    else:
        # Assume bigearthnet
        train_dataGen = BigEarthNet(data_dir, args.label_split_dir,
                                imgTransform=imgTransform,
                                label_type=label_type, threshold=args.threshold, subset="train",
                                use_s1=use_s1, use_s2=use_s2, use_RGB=args.use_RGB,
                                CLC_s=args.simple_scheme, data_size=args.data_size, sensor_type=args.sensor_type, use_fusion=args.use_fusion)

        val_dataGen = BigEarthNet(data_dir, args.label_split_dir,
                              imgTransform=imgTransform,
                              label_type=label_type, threshold=args.threshold, subset="val",
                              use_s1=use_s1, use_s2=use_s2, use_RGB=args.use_RGB,
                              CLC_s=args.simple_scheme, data_size=args.data_size, sensor_type=args.sensor_type, use_fusion=args.use_fusion)

        if args.eval:
            test_dataGen = BigEarthNet(data_dir, args.label_split_dir,
                                   imgTransform=imgTransform,
                                   label_type=label_type, threshold=args.threshold, subset="test",
                                   use_s1=use_s1, use_s2=use_s2, use_RGB=args.use_RGB,
                                   CLC_s=args.simple_scheme, sensor_type=args.sensor_type, use_fusion=args.use_fusion)
    
    # number of input channels
    n_inputs = train_dataGen.n_inputs 
    print('input channels =', n_inputs)
    wandb.config.update({"input_channels": n_inputs})

    # set up dataloaders
    train_data_loader = DataLoader(train_dataGen, 
                                   batch_size=args.batch_size, 
                                   num_workers=args.num_workers, 
                                   shuffle=True, 
                                   pin_memory=True)
    val_data_loader = DataLoader(val_dataGen, 
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 shuffle=False, 
                                 pin_memory=True)

    if args.eval:
        test_data_loader = DataLoader(test_dataGen,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False,
                                  pin_memory=True)

# -------------------------------- ML setup
    # cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    # define number of classes
    if dataset == 'sen12ms':
        if args.simple_scheme:
            numCls = 10
            ORG_LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        else:
            numCls = 17
            ORG_LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                          '11', '12', '13', '14', '15', '16', '17']
    else:
        if args.simple_scheme:
            numCls = 19
            ORG_LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                          '11', '12', '13', '14', '15', '16', '17', '18', '19']
        else:
            numCls = 43
            ORG_LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                          '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                          '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                          '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
                          '41', '42', '43']
    
    print('num_class: ', numCls)
    wandb.config.update({"n_class": numCls})

    # define model
    if args.model == 'VGG16':
        model = VGG16(n_inputs, numCls)
    elif args.model == 'VGG19':
        model = VGG19(n_inputs, numCls)
    elif args.model == 'Supervised':
        model = ResNet50(n_inputs, numCls)
    elif args.model == 'Supervised_1x1':
        model = ResNet50_1x1(n_inputs, numCls)
    elif args.model == 'ResNet101':
        model = ResNet101(n_inputs, numCls)
    elif args.model == 'ResNet152':
        model = ResNet152(n_inputs, numCls)
    elif args.model == 'DenseNet121':
        model = DenseNet121(n_inputs, numCls)
    elif args.model == 'DenseNet161':
        model = DenseNet161(n_inputs, numCls)
    elif args.model == 'DenseNet169':
        model = DenseNet169(n_inputs, numCls)
    elif args.model == 'DenseNet201':
        model = DenseNet201(n_inputs, numCls)
    # finetune moco pre-trained model
    elif args.model.startswith("Moco"):
        pt_path = os.path.join(args.pt_dir, f"{args.pt_name}.pth")
        print(pt_path)
        assert os.path.exists(pt_path)
        if args.model == 'Moco':
            print("transfer backbone weights but no conv 1x1 input module")
            model = Moco(torch.load(pt_path), n_inputs, numCls)
        elif args.model == 'Moco_1x1':
            print("transfer backbone weights and input module weights")
            model = Moco_1x1(torch.load(pt_path), n_inputs, numCls)
        elif args.model == 'Moco_1x1RND':
            print("transfer backbone weights but initialize input module random with random weights")
            model = Moco_1x1(torch.load(pt_path), n_inputs, numCls)
        else:  # Assume Moco2 at present
            raise NameError("no model")
    else:
        raise NameError("no model")

    print(model)

    # move model to GPU if is available
    if use_cuda:
        model = model.cuda() 

    # define loss function
    if label_type == 'multi_label':
        lossfunc = torch.nn.BCEWithLogitsLoss()
    else:
        lossfunc = torch.nn.CrossEntropyLoss()

    
    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    best_acc = 0
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            checkpoint_nm = os.path.basename(args.resume)
            sv_name = checkpoint_nm.split('_')[0] + '_' + checkpoint_nm.split('_')[1]
            print('saving file name is ', sv_name)

            if checkpoint['epoch'] > start_epoch:
                start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # set up tensorboard logging
    # train_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'training'))
    # val_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'val'))


# ----------------------------- executing Train/Val. 
    # train network
    # wandb.watch(model, log="all")

    scheduler = None
    if args.use_lr_step:
        # Ex: If initial Lr is 0.0001, step size is 25, and gamma is 0.1, then lr will be changed for every 20 steps
        # 0.0001 - first 25 epochs
        # 0.00001 - 25 to 50 epochs
        # 0.000001 - 50 to 75 epochs
        # 0.0000001 - 75 to 100 epochs
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_step_gamma)

    for epoch in range(start_epoch, args.epochs):
        if args.use_lr_step:
            scheduler.step()
            print('Epoch {}/{} lr: {}'.format(epoch, args.epochs - 1, optimizer.param_groups[0]['lr']))
        else:
            print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 25)

        train(train_data_loader, model, optimizer, lossfunc, label_type, epoch, use_cuda)
        micro_f1 = val(val_data_loader, model, optimizer, label_type, epoch, use_cuda)

        is_best_acc = micro_f1 > best_acc
        best_acc = max(best_acc, micro_f1)

        save_checkpoint({
            'epoch': epoch,
            'arch': args.model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_prec': best_acc
            }, is_best_acc, sv_name)

        wandb.log({'epoch': epoch, 'micro_f1': micro_f1})

    print("=============")
    print("done training")
    print("=============")

    if args.eval:
        eval(test_data_loader, model, label_type, numCls, use_cuda, ORG_LABELS)

def eval(test_data_loader, model, label_type, numCls, use_cuda, ORG_LABELS):

    model.eval()
    # define metrics
    prec_score_ = Precision_score()
    recal_score_ = Recall_score()
    f1_score_ = F1_score()
    f2_score_ = F2_score()
    hamming_loss_ = Hamming_loss()
    subset_acc_ = Subset_accuracy()
    acc_score_ = Accuracy_score()  # from original script, not recommeded, seems not correct
    one_err_ = One_error()
    coverage_err_ = Coverage_error()
    rank_loss_ = Ranking_loss()
    labelAvgPrec_score_ = LabelAvgPrec_score()

    calssification_report_ = calssification_report(ORG_LABELS)

    # -------------------------------- prediction
    y_true = []
    predicted_probs = []

    pred_dic = {}

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_data_loader, desc="test")):

            # unpack sample
            bands = data["image"]
            labels = data["label"]

            # move data to gpu if model is on gpu
            if use_cuda:
                bands = bands.to(torch.device("cuda"))
                # labels = labels.to(torch.device("cuda"))

            # forward pass
            logits = model(bands)

            # convert logits to probabilies
            if label_type == 'multi_label':
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                sm = torch.nn.Softmax(dim=1)
                probs = sm(logits).cpu().numpy()

            labels = labels.cpu().numpy()  # keep true & pred label at same loc.
            predicted_probs += list(probs)
            y_true += list(labels)

            if args.output_pred:
                # Cache the y_true and y_prediction in a dictionary for analysis
                for j in range(len(data['id'])):
                    pred_dic[data['id'][j]] = {'true': str(list(list(labels)[j])),
                                               'prediction': str(list(list(probs)[j]))
                                               }

    if args.output_pred:
        # Store the  y_true and y_prediction in a json file under checkpoint folder.
        # This file can be viewed under Files tab in wandb dashboard for a run
        fileout = f"{checkpoint_dir}/{sv_name_eval}_{args.model}_{label_type}.json"
        with open(fileout,'w') as fp:
            json.dump(pred_dic, fp)

    predicted_probs = np.asarray(predicted_probs)
    # convert predicted probabilities into one/multi-hot labels
    if label_type == 'multi_label':
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)
    else:
        loc = np.argmax(predicted_probs, axis=-1)
        y_predicted = np.zeros_like(predicted_probs).astype(np.float32)
        for i in range(len(loc)):
            y_predicted[i, loc[i]] = 1

    y_true = np.asarray(y_true)

    # --------------------------- evaluation with metrics
    # general
    macro_f1, micro_f1, sample_f1 = f1_score_(y_predicted, y_true)
    macro_f2, micro_f2, sample_f2 = f2_score_(y_predicted, y_true)
    macro_prec, micro_prec, sample_prec = prec_score_(y_predicted, y_true)
    macro_rec, micro_rec, sample_rec = recal_score_(y_predicted, y_true)
    hamming_loss = hamming_loss_(y_predicted, y_true)
    subset_acc = subset_acc_(y_predicted, y_true)
    macro_acc, micro_acc, sample_acc = acc_score_(y_predicted, y_true)
    # ranking-based
    one_error = one_err_(predicted_probs, y_true)
    coverage_error = coverage_err_(predicted_probs, y_true)
    rank_loss = rank_loss_(predicted_probs, y_true)
    labelAvgPrec = labelAvgPrec_score_(predicted_probs, y_true)

    cls_report = calssification_report_(y_predicted, y_true)

    if label_type == 'multi_label':
        [conf_mat, cls_acc, aa] = multi_conf_mat(y_predicted, y_true, n_classes=numCls)
        # the results derived from multilabel confusion matrix are not recommended to use
        oa = OA_multi(y_predicted, y_true)
        # this oa can be Jaccard index

        info = {
            "macroPrec": macro_prec,
            "microPrec": micro_prec,
            "samplePrec": sample_prec,
            "macroRec": macro_rec,
            "microRec": micro_rec,
            "sampleRec": sample_rec,
            "macroF1": macro_f1,
            "microF1": micro_f1,
            "sampleF1": sample_f1,
            "macroF2": macro_f2,
            "microF2": micro_f2,
            "sampleF2": sample_f2,
            "HammingLoss": hamming_loss,
            "subsetAcc": subset_acc,
            "macroAcc": macro_acc,
            "microAcc": micro_acc,
            "sampleAcc": sample_acc,
            "oneError": one_error,
            "coverageError": coverage_error,
            "rankLoss": rank_loss,
            "labelAvgPrec": labelAvgPrec,
            "clsReport": cls_report,
            "multilabel_conf_mat": conf_mat,
            "class-wise Acc": cls_acc,
            "AverageAcc": aa,
            "OverallAcc": oa}

    else:
        conf_mat = conf_mat_nor(y_predicted, y_true, n_classes=numCls)
        aa = get_AA(y_predicted, y_true, n_classes=numCls)  # average accuracy, \
        # zero-sample classes are not excluded

        info = {
            "macroPrec": macro_prec,
            "microPrec": micro_prec,
            "samplePrec": sample_prec,
            "macroRec": macro_rec,
            "microRec": micro_rec,
            "sampleRec": sample_rec,
            "macroF1": macro_f1,
            "microF1": micro_f1,
            "sampleF1": sample_f1,
            "macroF2": macro_f2,
            "microF2": micro_f2,
            "sampleF2": sample_f2,
            "HammingLoss": hamming_loss,
            "subsetAcc": subset_acc,
            "macroAcc": macro_acc,
            "microAcc": micro_acc,
            "sampleAcc": sample_acc,
            "oneError": one_error,
            "coverageError": coverage_error,
            "rankLoss": rank_loss,
            "labelAvgPrec": labelAvgPrec,
            "clsReport": cls_report,
            "conf_mat": conf_mat,
            "AverageAcc": aa}

    wandb.run.summary.update(info)
    print(model)
    print("saving metrics...")
    # pkl.dump(info, open("test_scores.pkl", "wb"))


def train(trainloader, model, optimizer, lossfunc, label_type, epoch, use_cuda):

    lossTracker = MetricTracker()

    # set model to train mode
    model.train()


    # main training loop
    for idx, data in enumerate(tqdm(trainloader, desc="training")):
        
        numSample = data["image"].size(0)
        
        # unpack sample
        bands = data["image"]
        if label_type == 'multi_label':
            labels = data["label"]
        else:
           labels = (torch.max(data["label"], 1)[1]).type(torch.long) 
               
        # move data to gpu if model is on gpu
        if use_cuda:
            bands = bands.to(torch.device("cuda"))
            labels = labels.to(torch.device("cuda"))
        
        # reset gradients
        optimizer.zero_grad()
        
        # forward pass
        logits = model(bands)
        loss = lossfunc(logits, labels)
        
        # backward pass
        loss.backward()
        optimizer.step()
        
        #
        lossTracker.update(loss.item(), numSample)

    # train_writer.add_scalar("loss", lossTracker.avg, epoch)
    wandb.log({'loss': lossTracker.avg, 'epoch': epoch})

    print('Train loss: {:.6f}'.format(lossTracker.avg))

    
def val(valloader, model, optimizer, label_type, epoch, use_cuda):

    prec_score_ = Precision_score()
    recal_score_ = Recall_score()
    f1_score_ = F1_score()
    f2_score_ = F2_score()
    hamming_loss_ = Hamming_loss()
    subset_acc_ = Subset_accuracy()
    acc_score_ = Accuracy_score()
    one_err_ = One_error()
    coverage_err_ = Coverage_error()
    rank_loss_ = Ranking_loss()
    labelAvgPrec_score_ = LabelAvgPrec_score()

    # set model to evaluation mode
    model.eval()
    
    # main validation loop
    y_true = []
    predicted_probs = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(valloader, desc="validation")):

            # unpack sample
            bands = data["image"]
            labels = data["label"]
    
            # move data to gpu if model is on gpu
            if use_cuda:
                bands = bands.to(torch.device("cuda"))
                #labels = labels.to(torch.device("cuda"))
            
            # forward pass 
            logits = model(bands)
            
            # convert logits to probabilies
            if label_type == 'multi_label':
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                sm = torch.nn.Softmax(dim=1)
                probs = sm(logits).cpu().numpy()
                  
            labels = labels.cpu().numpy() # keep true & pred label at same loc.
            predicted_probs += list(probs)
            y_true += list(labels)

    predicted_probs = np.asarray(predicted_probs)
    # convert predicted probabilities into one/multi-hot labels 
    if label_type == 'multi_label':
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)
    else:
        loc = np.argmax(predicted_probs, axis=-1)
        y_predicted = np.zeros_like(predicted_probs).astype(np.float32)
        for i in range(len(loc)):
            y_predicted[i,loc[i]] = 1
        
    y_true = np.asarray(y_true)
    

    macro_f1, micro_f1, sample_f1 = f1_score_(y_predicted, y_true)
    macro_f2, micro_f2, sample_f2 = f2_score_(y_predicted, y_true)
    macro_prec, micro_prec, sample_prec = prec_score_(y_predicted, y_true)
    macro_rec, micro_rec, sample_rec = recal_score_(y_predicted, y_true)
    hamming_loss = hamming_loss_(y_predicted, y_true)
    subset_acc = subset_acc_(y_predicted, y_true)
    macro_acc, micro_acc, sample_acc = acc_score_(y_predicted, y_true)

    # Note that below 4 ranking-based metrics are not applicable to single-label
    # (multi-class) classification, but they will still show the scores during 
    # validation on tensorboard
    one_error = one_err_(predicted_probs, y_true)
    coverage_error = coverage_err_(predicted_probs, y_true)
    rank_loss = rank_loss_(predicted_probs, y_true)
    labelAvgPrec = labelAvgPrec_score_(predicted_probs, y_true)

    info = {
            "macroPrec" : macro_prec,
            "microPrec" : micro_prec,
            "samplePrec" : sample_prec,
            "macroRec" : macro_rec,
            "microRec" : micro_rec,
            "sampleRec" : sample_rec,
            "macroF1" : macro_f1,
            "microF1" : micro_f1,
            "sampleF1" : sample_f1,
            "macroF2" : macro_f2,
            "microF2" : micro_f2,
            "sampleF2" : sample_f2,
            "HammingLoss" : hamming_loss,
            "subsetAcc" : subset_acc,
            "macroAcc" : macro_acc,
            "microAcc" : micro_acc,
            "sampleAcc" : sample_acc,
            "oneError" : one_error,
            "coverageError" : coverage_error,
            "rankLoss" : rank_loss,
            "labelAvgPrec" : labelAvgPrec
            }

    wandb.run.summary.update(info)
    for tag, value in info.items():
        wandb.log({tag: value, 'epoch': epoch})
        # val_writer.add_scalar(tag, value, epoch)

    print('Validation microPrec: {:.6f} microF1: {:.6f} sampleF1: {:.6f} microF2: {:.6f} sampleF2: {:.6f}'.format(
            micro_prec,
            micro_f1,
            sample_f1,
            micro_f2,
            sample_f2
            ))
    return micro_f1




if __name__ == "__main__":
    main()
    
    