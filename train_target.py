import argparse
import datetime 

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='0,1')
parser.add_argument('--model-file', type=str, default=
                    '/users/scratch/baryaacovi-2024-06-01/projects/SFDA-CBMT/logs_train/Domain3/20240704_014607.138681/checkpoint_175.pth.tar'
                    )
parser.add_argument('--only-pseudo', type=bool, default=False)
parser.add_argument('--pseudo-path', type=str, default=f'./exp/{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}/pseudo_label_cbmt.npz')
parser.add_argument('--pkl-save-path', type=str, default=f'./exp/{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}/pseudo_label_cbmt.pkl')
parser.add_argument('--best-n', type=int, default=70)
parser.add_argument('--order-repeat', type=int, default=0)
parser.add_argument('--pkl-path', type=str, default=
                    '/users/scratch/baryaacovi-2024-06-01/projects/SFDA-DPL/results/sam_ranking/Domain2_samvdpl_CBMT.pkl'
                    )
parser.add_argument('--sam-every', type=int, default=2)
parser.add_argument('--use-sam', type=bool, default=True)
parser.add_argument('--use-gt', type=bool, default=False)
parser.add_argument('--use-top-n', type=bool, default=True)

parser.add_argument('--model', type=str, default='Deeplab', help='Deeplab')
parser.add_argument('--out-stride', type=int, default=16)
parser.add_argument('--sync-bn', type=bool, default=True)
parser.add_argument('--freeze-bn', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=20)

parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr-decrease-rate', type=float, default=0.9, help='ratio multiplied to initial lr')
parser.add_argument('--lr-decrease-epoch', type=int, default=1, help='interval epoch number for lr decrease')

parser.add_argument('--data-dir', default='../datasets/Fundus')
parser.add_argument('--dataset', type=str, default='Domain2')
parser.add_argument('--model-source', type=str, default='Domain3')
parser.add_argument('--batch-size', type=int, default=8)

parser.add_argument('--model-ema-rate', type=float, default=0.98)
parser.add_argument('--pseudo-label-threshold', type=float, default=0.75)
parser.add_argument('--mean-loss-calc-bound-ratio', type=float, default=0.2)

args = parser.parse_args()
print("GPU: ", args.gpu)
import os
if not os.path.exists(os.path.dirname(args.pseudo_path)):
    os.makedirs(os.path.dirname(args.pseudo_path))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import os.path as osp

import numpy as np
import torch.nn.functional as F

import torch
from torch.autograd import Variable
import tqdm
from torch.utils.data import DataLoader
from dataloaders import fundus_dataloader
from dataloaders import custom_transforms as trans
from torchvision import transforms
# from scipy.misc import imsave
from matplotlib.pyplot import imsave
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import pytz
import networks.deeplabv3 as netd
import cv2
import torch.backends.cudnn as cudnn
import random
import glob
import sys
from generate_SAM_ranking import generate_sam_score

seed = 42
savefig = False
get_hd = True
model_save = True
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def soft_label_to_hard(soft_pls, pseudo_label_threshold):
    pseudo_labels = torch.zeros(soft_pls.size())
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[soft_pls > pseudo_label_threshold] = 1
    pseudo_labels[soft_pls <= pseudo_label_threshold] = 0

    return pseudo_labels


def init_feature_pred_bank(model, loader, top_images = None):
    feature_bank = {}
    pred_bank = {}

    model.eval()

    with torch.no_grad():
        for sample in loader:
            data = sample['image']
            img_name = sample['img_name']
            data = data.cuda()

            pred, feat = model(data)
            pred = torch.sigmoid(pred)

            for i in range(data.size(0)):
                if top_images == None or img_name[i] in top_images:
                    feature_bank[img_name[i]] = feat[i].detach().clone()
                    pred_bank[img_name[i]] = pred[i].detach().clone()

    model.train()

    return feature_bank, pred_bank

def adapt_epoch(model_t, model_s, optim, train_loader, args, feature_bank, pred_bank, loss_weight=None, top_images=None):
    for sample_w, sample_s in train_loader:
        imgs_w = sample_w['image']
        imgs_s = sample_s['image']
        img_name = sample_w['img_name']
        if torch.cuda.is_available():
            imgs_w = imgs_w.cuda()
            imgs_s = imgs_s.cuda()

        # model predict
        predictions_stu_s, features_stu_s = model_s(imgs_s)
        with torch.no_grad():
            predictions_tea_w, features_tea_w = model_t(imgs_w)

        predictions_stu_s_sigmoid = torch.sigmoid(predictions_stu_s)
        predictions_tea_w_sigmoid = torch.sigmoid(predictions_tea_w).detach()
        
        # get hard pseudo label
        pseudo_labels = soft_label_to_hard(predictions_tea_w_sigmoid, args.pseudo_label_threshold)

        bceloss = torch.nn.BCELoss(reduction='none')
        loss_seg_pixel = bceloss(predictions_stu_s_sigmoid, pseudo_labels.cuda())
        
        mean_loss_weight_mask = torch.ones(pseudo_labels.size()).cuda()
        mean_loss_weight_mask[:, 0, ...][pseudo_labels[:, 0, ...] == 0] = loss_weight
        loss_mask = mean_loss_weight_mask
        
        sum_before_removal_cup = torch.sum(loss_mask)
        if top_images is not None:
            for i in range(pseudo_labels.size(0)):
                if img_name[i] not in top_images:
                    loss_mask[i,0] = 0

        loss = torch.sum(loss_seg_pixel * loss_mask) / torch.sum(loss_mask)

        loss.backward()
        optim.step()
        optim.zero_grad()

        # update teacher
        for param_s, param_t in zip(model_s.parameters(), model_t.parameters()):
            param_t.data = param_t.data.clone() * args.model_ema_rate + param_s.data.clone() * (1.0 - args.model_ema_rate)

        # update feature/pred bank
        for idx in range(len(img_name)):
            feature_bank[img_name[idx]] = features_tea_w[idx].detach().clone()
            pred_bank[img_name[idx]] = predictions_tea_w_sigmoid[idx].detach().clone()


def eval(model, data_loader, save_pseudo = False, save_pkl = False, num_epoch = 0, pkl_save_path = None):
    model.eval()

    val_dice = {'cup': np.array([]), 'disc': np.array([])}
    val_assd = {'cup': np.array([]), 'disc': np.array([])}

    pseudo_label_dic = {}
    dice_cup_dict = {}
    save_list = []
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            data = sample['image']
            target_map = sample['label']
            img_name = sample['img_name']
            
            data = data.cuda()
            predictions, _ = model(data)

            dice_cup, dice_disc = dice_coeff_2label(predictions, target_map)
            val_dice['cup'] = np.append(val_dice['cup'], dice_cup)
            val_dice['disc'] = np.append(val_dice['disc'], dice_disc)

            assd = assd_compute(predictions, target_map)
            val_assd['cup'] = np.append(val_assd['cup'], assd[:, 0])
            val_assd['disc'] = np.append(val_assd['disc'], assd[:, 1])
            
            pred = torch.sigmoid(predictions)
            pred = pred.data.cpu()
            pred[pred > 0.75] = 1
            pred[pred <= 0.75] = 0
            for i in range(len(img_name)):
                if save_pseudo:
                    pseudo_label_dic[img_name[i]] = pred[i].detach().cpu().numpy()
                    dice_cup_dict[img_name[i]] = dice_cup[i]
                if save_pkl:
                    save_list.append((dice_cup[i],dice_cup[i],img_name[i], dice_disc[i]))


        avg_dice = [0.0, 0.0, 0.0, 0.0]
        std_dice = [0.0, 0.0, 0.0, 0.0]
        avg_assd = [0.0, 0.0, 0.0, 0.0]
        std_assd = [0.0, 0.0, 0.0, 0.0]
        avg_dice[0] = np.mean(val_dice['cup'])
        avg_dice[1] = np.mean(val_dice['disc'])
        std_dice[0] = np.std(val_dice['cup'])
        std_dice[1] = np.std(val_dice['disc'])
        val_assd['cup'] = np.delete(val_assd['cup'], np.where(val_assd['cup'] == -1))
        val_assd['disc'] = np.delete(val_assd['disc'], np.where(val_assd['disc'] == -1))
        avg_assd[0] = np.mean(val_assd['cup'])
        avg_assd[1] = np.mean(val_assd['disc'])
        std_assd[0] = np.std(val_assd['cup'])
        std_assd[1] = np.std(val_assd['disc'])
        
        
        if save_pseudo:
            p = args.pseudo_path
            print("Save pseudo labels to: ", p)
            np.savez(p, pseudo_label_dic, dice_cup_dict)
        if save_pkl:
            save_list = sorted(save_list, key=lambda x: x[0])
            if pkl_save_path is None:
                pkl_p = args.pkl_save_path
            else:
                pkl_p = pkl_save_path
            # put save_list in a pkl file
            print("Save pkl to: ", pkl_p)
            with open(pkl_p, "wb") as f:
                import pickle
                pickle.dump(save_list, f)
            

    model.train()
    if save_pseudo:
        return avg_dice, std_dice, avg_assd, std_assd, pseudo_label_dic, dice_cup_dict
    return avg_dice, std_dice, avg_assd, std_assd


def main():
    print("Starting...")
    now = datetime.now()
    here = osp.dirname(osp.abspath(__file__))
    args.out = osp.join(here, 'logs_target', args.dataset, now.strftime('%Y%m%d_%H%M%S.%f'))
    if not osp.exists(args.out):
        os.makedirs(args.out)
    args.out_file = open(osp.join(args.out, now.strftime('%Y%m%d_%H%M%S.%f')+'.txt'), 'w')
    args.out_file.write(' '.join(sys.argv) + '\n')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    # dataset
    composed_transforms_train = transforms.Compose([
        trans.Resize(512),
        trans.add_salt_pepper_noise(),
        trans.adjust_light(),
        trans.eraser(),
        trans.Normalize_tf(),
        trans.ToTensor()
    ])
    composed_transforms_test = transforms.Compose([
        trans.Resize(512),
        trans.Normalize_tf(),
        trans.ToTensor()
    ])
    
    pkl_path = args.pkl_path
    n = args.best_n
    
    print(args.lr)
    print(args.model_file)
    print(pkl_path)
    print("Output folder: ", args.out)
    print("Model ema rate: ", args.model_ema_rate)
    
    top_images = None
    
    top_images, the_best_ones = get_top_images(pkl_path, n, 
                                # threshold=0.4,
                                # repeat=args.order_repeat,
                                )
    
    if not args.use_top_n:
        print("Setting top images to None")
        top_images, the_best_ones = None, None
        

    train_loader, train_loader_weak, test_loader = get_data_loaders(composed_transforms_train, composed_transforms_test, top_images, the_best_ones=the_best_ones)

    # model
    model_s = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn,
                           freeze_bn=args.freeze_bn)
    model_t = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn,
                           freeze_bn=args.freeze_bn)


    if torch.cuda.is_available():
        model_s = model_s.cuda(0)
        model_t = model_t.cuda(0)
    log_str = '==> Loading %s model file: %s' % (model_s.__class__.__name__, args.model_file)
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    checkpoint = torch.load(args.model_file)
    model_s.load_state_dict(checkpoint['model_state_dict'])
    model_t.load_state_dict(checkpoint['model_state_dict'])

    # if (args.gpu).find(',') != -1:
    #     model_s = torch.nn.DataParallel(model_s, device_ids=[0, 1])
    #     model_t = torch.nn.DataParallel(model_t, device_ids=[0, 1])

    optim = torch.optim.Adam(model_s.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_decrease_epoch, gamma=args.lr_decrease_rate)

    model_s.train()
    model_t.train()
    for param in model_t.parameters():
        param.requires_grad = False


    feature_bank, pred_bank = init_feature_pred_bank(model_s, train_loader_weak, top_images=top_images)


    if args.only_pseudo:
        avg_dice, std_dice, avg_assd, std_assd,_,_ = eval(model_t, train_loader_weak, save_pseudo=True, save_pkl=True)
        return
    avg_dice, std_dice, avg_assd, std_assd = eval(model_t, test_loader)
    log_str = ("initial dice: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f, assd: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f" % (
            avg_dice[0], std_dice[0], avg_dice[1], std_dice[1], (avg_dice[0] + avg_dice[1]) / 2.0,
            avg_assd[0], std_assd[0], avg_assd[1], std_assd[1], (avg_assd[0] + avg_assd[1]) / 2.0))
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    thresh = 0.4
    for epoch in range(args.epoch):

        log_str = '\nepoch {}/{}:'.format(epoch+1, args.epoch)
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        not_cup_loss_sum = torch.FloatTensor([0]).cuda()
        cup_loss_sum = torch.FloatTensor([0]).cuda()
        not_cup_loss_num = 0
        cup_loss_num = 0
        lower_bound = args.pseudo_label_threshold * args.mean_loss_calc_bound_ratio
        upper_bound = 1 - ((1 - args.pseudo_label_threshold) * args.mean_loss_calc_bound_ratio)
        for pred_i in pred_bank.values():
            not_cup_loss_sum += torch.sum(
                -torch.log(1 - pred_i[0, ...][(pred_i[0, ...] < args.pseudo_label_threshold) * (pred_i[0, ...] > lower_bound)]))
            not_cup_loss_num += torch.sum((pred_i[0, ...] < args.pseudo_label_threshold) * (pred_i[0, ...] > lower_bound))
            cup_loss_sum += torch.sum(-torch.log(pred_i[0, ...][(pred_i[0, ...] > args.pseudo_label_threshold) * (pred_i[0, ...] < upper_bound)]))
            cup_loss_num += torch.sum((pred_i[0, ...] > args.pseudo_label_threshold) * (pred_i[0, ...] < upper_bound))
        loss_weight = (cup_loss_sum.item() / cup_loss_num) / (not_cup_loss_sum.item() / not_cup_loss_num)

        adapt_epoch(model_t, model_s, optim, train_loader, args, feature_bank, pred_bank, loss_weight=loss_weight, top_images=top_images)

        scheduler.step()
              
        avg_dice, std_dice, avg_assd, std_assd = eval(model_t, test_loader)
        log_str = ("teacher dice: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f, assd: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f" % (
            avg_dice[0], std_dice[0], avg_dice[1], std_dice[1], (avg_dice[0] + avg_dice[1]) / 2.0,
            avg_assd[0], std_assd[0], avg_assd[1], std_assd[1], (avg_assd[0] + avg_assd[1]) / 2.0))
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        avg_dice, std_dice, avg_assd, std_assd = eval(model_s, test_loader)
        log_str = ("student dice: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f, assd: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f" % (
                avg_dice[0], std_dice[0], avg_dice[1], std_dice[1], (avg_dice[0] + avg_dice[1]) / 2.0,
                avg_assd[0], std_assd[0], avg_assd[1], std_assd[1], (avg_assd[0] + avg_assd[1]) / 2.0))
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        
        if epoch > 0 and epoch % args.sam_every == 0 and (args.use_sam or args.use_gt):
            if thresh < 0.60:
                thresh+=0.01
                print("Current threshold: ", thresh)
            print("updating DL")
            if n <= 90:
                n += 5
                if n%8 == 1:
                    n = n + 1
                    
            # if n < 46:
            #     n += 2
            pkl_save = args.pseudo_path.replace(".npz", f"_e{epoch}_{n}_tmp.pkl")
            
            dataset_train_weak = fundus_dataloader.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset,
                                                            split='train/ROIs',
                                                            transform=composed_transforms_test, 
                                                            # only_names = top_images,
                                                            # order_names = top_images
                                                            )
            train_loader_weak = DataLoader(dataset_train_weak, batch_size=args.batch_size, shuffle=False, num_workers=2)
            
            # generate PL
            if args.use_sam:
                avg_dice, std_dice, avg_assd, std_assd,pseudo_dic, dice_cup_dic = eval(model_t, train_loader_weak, save_pseudo=True,num_epoch = epoch, save_pkl=False, pkl_save_path = pkl_save)
                train_loader_weak_sam = DataLoader(dataset_train_weak, batch_size=1, shuffle=False, num_workers=2)
                print("START CREATING SAM SCORE")
                print("Will save pkl to: ", pkl_save)
                generate_sam_score(train_loader_weak_sam,pseudo_dic, dice_cup_dic, pkl_save, epoch)
            elif args.use_gt:
                print("START CREATING GROUND TRUTH SCORE")
                avg_dice, std_dice, avg_assd, std_assd,pseudo_dic, dice_cup_dic = eval(model_t, train_loader_weak, save_pseudo=True,num_epoch = epoch, save_pkl=True, pkl_save_path = pkl_save)
                
            top_images, the_best_ones = get_top_images(pkl_save, n, repeat=args.order_repeat
                                        # threshold=0.5
                                        )
                
            train_loader, train_loader_weak, test_loader = get_data_loaders(composed_transforms_train, composed_transforms_test, top_images, the_best_ones)
            rm_list = []
            for name in pred_bank.keys():
                if name not in top_images:
                    rm_list.append(name)
            for name in rm_list:
                pred_bank.pop(name)
                feature_bank.pop(name)
                
            feature_bank, pred_bank = init_feature_pred_bank(model_s, train_loader_weak, top_images=top_images)
                
    torch.save({'model_state_dict': model_t.state_dict()}, args.out + '/after_adaptation.pth.tar')

def get_data_loaders(composed_transforms_train, composed_transforms_test, top_images, the_best_ones = None):
    dataset_train = fundus_dataloader.FundusSegmentation_2transform(base_dir=args.data_dir, dataset=args.dataset,
                                                                    split='train/ROIs',
                                                                    transform_weak=composed_transforms_test,
                                                                    transform_strong=composed_transforms_train,
                                                                    # only_names = top_images,
                                                                    order_names = the_best_ones
                                                                    )
    dataset_train_weak = fundus_dataloader.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset,
                                                              split='train/ROIs',
                                                              transform=composed_transforms_test, 
                                                            #   only_names = top_images,
                                                            order_names = the_best_ones
                                                              )
    dataset_test = fundus_dataloader.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test/ROIs',
                                         transform=composed_transforms_test)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    train_loader_weak = DataLoader(dataset_train_weak, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2)
    return train_loader,train_loader_weak,test_loader

def get_top_images(pkl_path, n, threshold =-1, repeat=0):
    repeat = repeat
    print("Reading pkl file: ", pkl_path)
    with open(pkl_path, "rb") as f:
        import pickle
        all_images_tuples = pickle.load(f)
        # from random import shuffle
        # shuffle(all_images_tuples)
        
        if threshold > 0:
            best_n = [e for e in all_images_tuples if e[0] > threshold]
            n = len(best_n)
            # print("Computed n value: ", n)
            if n % 8 == 1:
                n = n -1
                best_n.remove(best_n[0])
            # print("Computed n value: ", n)
        else:
            best_n = all_images_tuples[-n:]
        if repeat > 0:
            repeat_names = [e[2] for e in all_images_tuples[-repeat:]]
        else:
            repeat_names = None
        
        top_images = [e[2] for e in best_n]
        dice_gt_best_n = [e[1] for e in best_n]
        print(f"FIRST {n} DICE - we take these")
        print(sum(dice_gt_best_n)/len(dice_gt_best_n))
        # print(top_images)
        print("Dice: ", dice_gt_best_n)
        
        worst_n = all_images_tuples[:(len(all_images_tuples)-n)]
        print(f"LAST {99-n} DICE - these are ignored")
        dice_gt_worst_n = [e[1] for e in worst_n]
        if len(dice_gt_worst_n) > 0:
            print(sum(dice_gt_worst_n)/len(dice_gt_worst_n))
        print(dice_gt_worst_n)
        print("Worst n names: ", [e[2] for e in worst_n])
        # print("Len of top images: ", len(top_images))
        
        print()
        sorted_by_disc = sorted(all_images_tuples, key=lambda x: x[1])
        print("Bottom five of discs")
        print([sorted_by_disc[i] for i in range(5)])
        return top_images, repeat_names


if __name__ == '__main__':
    main()