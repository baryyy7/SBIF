import argparse
import os
import os.path as osp
import torch.nn.functional as F
import torch.nn as nn
from networks.deeplabv3 import *

# import matplotlib
# %matplotlib.use('TkAgg')
# %matplotlib inline
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from tqdm import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
# from scipy.misc import imsave
from matplotlib.pyplot import imsave
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import pytz
import networks.deeplabv3 as netd
# import networks.deeplabv3_eval as netd_eval
import cv2
import torch.backends.cudnn as cudnn
import random
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.patches as mpatches
# %matplotlib inline
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import masks_to_boxes
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("./segment-anything")
from segment_anything import sam_model_registry, SamPredictor
from sam_wrapper.sam_prompter import SamPrompterAll, get_sam_on_boxes
# import networks.deeplabv3_eval as netd_eval

def show_mask(mask, ax, random_color=False, gt_color=False, col = '' ,alpha=0.6):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif gt_color:
        color = np.array([170/255, 255/255, 0/255, alpha])
    elif col == '':
        color = np.array([30/255, 144/255, 255/255, alpha])
    elif col == 'red':
        color = np.array([255/255, 0/255, 0/255, alpha])
    elif col == 'green':
        color = np.array([0/255, 255/255, 0/255, alpha])
    elif col == 'blue':
        color = np.array([0/255, 0/255, 255/255, alpha])
    elif col == 'yellow':
        color = np.array([255/255, 255/255, 0/255, alpha])
    elif col == 'purple':
        color = np.array([255/255, 0/255, 255/255, alpha])
    elif col == 'cyan':
        color = np.array([0/255, 255/255, 255/255, alpha])
    elif col == 'orange':
        color = np.array([255/255, 165/255, 0/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=111):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=0.35)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=0.35)   
    
def show_box(box, ax, edgecolor='green'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))    

def show_sam_dpl(view_img, img_name, pseudo_boxes, target_masks,cup_masks_dpl, cup_masks_sam,points_cup, labels_cup, sam_dice_coef, dpl_dict_coef):
    fig,axes = plt.subplots(1,2,figsize=(30,10))
    subtitle = f"Image: {img_name[0]}, SAM: {round(sam_dice_coef[0], 4) * 100}, DPL: {round(dpl_dict_coef[0], 4) * 100}"
    fig.suptitle(subtitle)
    
    axes[0].imshow(view_img)
    show_mask(cup_masks_dpl[0][0], axes[0], alpha=0.35)
    # show_mask(target_masks[0][0].cpu().numpy(), axes[0], gt_color=True, alpha = 0.35)
    # show_mask(cup_masks_dpl[0][1].cpu().numpy(), axes[0], gt_color=True, alpha = 0.35)
    # show_box(pseudo_boxes[0], axes[0])
    
    axes[1].imshow(view_img)
    show_mask(cup_masks_sam[0], axes[1], alpha=0.35)
    show_mask(target_masks[0][0].cpu().numpy(), axes[1], gt_color=True, alpha = 0.35)
    show_box(pseudo_boxes[0], axes[1])
    if points_cup is not None and len(points_cup) > 0:
        show_points(points_cup, labels_cup, axes[1])
    
    
# 3. SAM
current_ranking = {}
predictor = None
def init_sam():
    global predictor
    print("Loading SAM...")
    sam_checkpoint = "./sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:1"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    print("Done loading SAM!")

seed = 3377
if True:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# def get_denoised_mask(data, img_name):
    
#     pseudo_label = [pseudo_label_dic.get(key) for key in img_name]
#     pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float().cuda()
#     return pseudo_label
    
#     prediction, _, feature = model(data)
#     prediction = torch.sigmoid(prediction)

#     pseudo_label = [pseudo_label_dic.get(key) for key in img_name]
#     proto_pseudo = [proto_pseudo_dic.get(key) for key in img_name]

#     # print(pseudo_label)
#     pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float().cuda()
#     uncertain_map = torch.from_numpy(np.asarray(uncertain_map)).float().cuda()
#     proto_pseudo = torch.from_numpy(np.asarray(proto_pseudo)).float().cuda()

#     target_0_obj = F.interpolate(pseudo_label[:,0:1,...], size=feature.size()[2:], mode='nearest')
#     target_1_obj = F.interpolate(pseudo_label[:, 1:, ...], size=feature.size()[2:], mode='nearest')
#     target_0_bck = 1.0 - target_0_obj;target_1_bck = 1.0 - target_1_obj

#     mask_0_obj = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
#     mask_0_bck = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
#     mask_1_obj = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
#     mask_1_bck = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
#     mask_0_obj[uncertain_map[:, 0:1, ...] < 0.05] = 1.0
#     mask_0_bck[uncertain_map[:, 0:1, ...] < 0.05] = 1.0
#     mask_1_obj[uncertain_map[:, 1:, ...] < 0.05] = 1.0
#     mask_1_bck[uncertain_map[:, 1:, ...] < 0.05] = 1.0
    
#     # confidence threshold
#     mask = torch.cat((mask_0_obj*pseudo_label[:,0:1,...]
#                     #   + mask_0_bck*(1.0-pseudo_label[:,0:1,...])
#                       , 
#                       mask_1_obj*pseudo_label[:,1:,...] 
#                     #   + mask_1_bck*(1.0-pseudo_label[:,1:,...])
#                       )
#                      , dim=1)
#     # mask = pseudo_label
    
#     mask_proto = torch.zeros([data.shape[0], 2, data.shape[2], data.shape[3]]).cuda()
#     mask_proto[pseudo_label==proto_pseudo] = 1.0

#     mask = mask*mask_proto
#     # print("MASK SHAPE", mask.shape)
#     return mask

@torch.no_grad()
def generate_sam_score(train_loader,pseudo_label_dic, dice_cup_dic, filename_pkl, epoch):
    if predictor is None:
        init_sam()
    def get_fixed_bbox(prediction, ):
        a = measure.label(prediction[0])
        # print(a.max())
        # print(a.min())
        # plt.imshow(view_img)
        colors = ["red", "green","blue","purple","pink","brown"]

        i = 0
        max_props = None
        max_box = 0
        # print(prediction.shape)
        # print(a.shape)
        for curim in regionprops(a):
            curim.area_bbox
            if max_box < curim.area_bbox:
                max_box = curim.area_bbox
                max_props = curim
            # minr, minc, maxr, maxc = curim.bbox
            # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
            #                          fill=False, edgecolor=colors[i], linewidth=2)
            # plt.gca().add_patch(rect)
            i += 1

        minr, minc, maxr, maxc = max_props.bbox
        return [minc, minr, maxc, maxr]
    # show_mask(prediction[0][0], plt.gca())

    # show_mask(target[0][0].cpu().numpy(), plt.gca(), gt_color=True, alpha=0.4)
    # show_mask(cup_pseudo_denoise, plt.gca(), col='green')
    # show_mask(pseudo_mask_pred[0], plt.gca(), col='cyan')
    # show_box(boxes[0], plt.gca())
    # show_box(pseudo_boxes[0], plt.gca(), edgecolor='red')
    # show_box(pseudo_boxes_pred[0], plt.gca(), edgecolor='cyan')
    # plt.show()
    
    show_vis = False
    show_vis_jacob = False
    max_idx = 0
    # torch.cuda.empty_cache()

    # print(np.__version__)
    # box 1 - disc, box 0 - cup
    pseudo_label_dic_sam = {}
    sam_points = 16
    sam_fix_points = 7
    fix_multi = 0.08
    max_dis = 0.5
    use_points = True
    use_boxes = True
    use_masks = False
    soft_mask = False
    initial_negative = False
    iterative_tries = 2
    dpl_avg = 5
    pseudo_ensemble = 3

    logits_sam_coef = 0.6

    # 8
    sam_points_cup = 10
    sam_points_cup_second = 4


    sam_neg = 0


    sam_buffer = False
    sam_radius_ratio = 0.75
    sam_radius_ratio_cup = 0.75
    sam_radius_ratio_cup_second = 0.4
    enable_inner_cup = False
    enable_inner_disc = True
    get_hd = True
    val_cup_dice = 0.0;val_disc_dice = 0.0;datanum_cnt = 0.0
    val_sam_cup_dice = 0.0;val_sam_disc_dice = 0.0
    val_sam_cup_pseudo_dice = 0.0;val_sam_disc_pseudo_dice = 0.0
    val_saminter_cup_dice = 0; val_samplus_cup_dice = 0
    val_saminter_disc_dice = 0; val_sam_disc_fix_dice = 0; val_saminter_disc_fix_dice = 0
    cup_hd = 0.0; disc_hd = 0.0;datanum_cnt_cup = 0.0;datanum_cnt_disc = 0.0
    val_sam_cup_logits_dice = 0.0; val_sam_cup_gt_dice = 0.0
    val_sam_cup_improve_dice_iter = [0.0 for _ in range(iterative_tries)]
    val_sam_cup_improve_dice_iter_buf = [0.0 for _ in range(iterative_tries)]
    val_sam_cup_ens = [0.0 for _ in range(3)]
    
    sam_vs_dpl_dice_pts = []
    gt_vs_dpl_dice_pts = []
    sam_vs_dpl_dice_pts_other = []
    sam_vs_dpl_dice_pts_third = []
    sam_dice_pts = []
    all_names = []

    for sample in tqdm(train_loader):
        data, target, img_name = sample['image'], sample['label'], sample['img_name']
        target_numpy = target.data.cpu()      
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # masks = get_denoised_mask(model, data, img_name).cpu().detach()
        
        
        # cup_dice = dice_coefficient_numpy(masks[:,0, ...], target_numpy[:, 0, ...])
        # disc_dice = dice_coefficient_numpy(masks[:,1, ...], target_numpy[:, 1, ...])
        
        # print("!!! DICE DPL - CUP", cup_dice)
        # print("!!! DICE DPL - DISC", disc_dice)
        
        
        pseudo_label = [pseudo_label_dic.get(key) for key in img_name]
        masks = torch.from_numpy(np.asarray(pseudo_label)).float()
            
        cup_dice = dice_coefficient_numpy(masks[:,0, ...], target_numpy[:, 0, ...])
        disc_dice = dice_coefficient_numpy(masks[:,1, ...], target_numpy[:, 1, ...])
        # print(f"IMG: {img_name} **DENOISED** CUP_DICE: {cup_dice} DISC_DICE: {disc_dice[0]}")

        try:
            pseudo_boxes = masks_to_boxes(masks[0]).cpu()
            # fixed_cup_box = get_fixed_bbox(masks[0])
            # for i in range(len(fixed_cup_box)):
            #     pseudo_boxes[0][i] = fixed_cup_box[i]
        except BaseException as err:
            print(f"## ERR - {img_name}")
            print(err)
            continue
                
            
        view_img = make_grid(
                        data[0, ...].clone().cpu().data, 1, normalize=True).detach().cpu().numpy()
        view_img = view_img.transpose(1,2,0)
        # print(view_img)
        view_img = (view_img * 255).astype(np.uint8)
        predictor.set_image(view_img)
        
        prompter_pseudo = SamPrompterAll(boxes=pseudo_boxes,sam_points=sam_points, sam_radius_ratio=sam_radius_ratio, sam_points_cup=sam_points_cup, sam_radius_ratio_cup=sam_radius_ratio_cup,
                                use_points=use_points, use_boxes=use_boxes, fix_multi=fix_multi, sam_fix_points=sam_fix_points, sam_neg=sam_neg, initial_negative = initial_negative,
                                enable_inner_cup = enable_inner_cup)
        
        points_cup_ps, points_disc_ps, labels_cup_ps, input_labels_ps, = prompter_pseudo.get_points_from_boxes_sam(radius_cup=sam_radius_ratio_cup_second, num_cup=sam_points_cup_second)
        
        # dice_masks, cup_masks, scores, logits = get_sam_on_boxes(view_img, predictor, pseudo_boxes, points=points_disc_ps, labels=input_labels_ps,points_cup=points_cup_ps, labels_cup = labels_cup_ps, box_buffer=sam_buffer,
        #                             multimask=False, use_points = use_points, use_boxes = use_boxes, use_masks = use_masks, logits_dpl = None)
    
        dice_masks, cup_masks, scores, logits = get_sam_on_boxes(view_img, predictor, pseudo_boxes, points=points_disc_ps, labels=input_labels_ps,points_cup=points_cup_ps, labels_cup = labels_cup_ps, box_buffer=sam_buffer,
                            multimask=False, use_points = use_points, use_boxes = use_boxes, use_masks = use_masks, logits_dpl = None, 
                            # rel_buffer=0.15, 
                            # buffer_type=0
                            rel_buffer=0.075, buffer_type=1
                            )
        

        
        dice_sam_cup = dice_coefficient_numpy( cup_masks, target_numpy[:, 0, ...])
        dice_sam_disc = dice_coefficient_numpy( dice_masks, target_numpy[:, 1, ...])
        dice_sam_vs_dpl = dice_coefficient_numpy( cup_masks, masks[:,0, ...])
        
        # print("!!! DICE DPL PSEUDO - CUP", cup_dice)
        # print("!!! DICE SAM - CUP", dice_sam_cup)
        # print("!!! DICE SAM VS DPL", dice_sam_vs_dpl)
        
        gt_vs_dpl_dice_pts.append(cup_dice[0])
        if cup_masks.sum() / (512 * 512) > 0.3:
            print("Appending 0, cup too large")
            sam_vs_dpl_dice_pts.append(0)
        else:
            sam_vs_dpl_dice_pts.append(dice_sam_vs_dpl[0])
        
        sam_dice_pts.append(dice_sam_cup[0])
        all_names.append(img_name[0])
        
        # show_sam_dpl(view_img, img_name, pseudo_boxes, target_numpy, masks, cup_masks ,points_cup_ps, labels_cup_ps, dice_sam_cup, cup_dice)
        # plt.savefig(f'./1111111111111{img_name[0]}.png')
        
        points_cup_ps, points_disc_ps, labels_cup_ps, input_labels_ps, = prompter_pseudo.get_points_from_boxes_sam()
        dice_masks, cup_masks, scores, logits = get_sam_on_boxes(view_img, predictor, pseudo_boxes, points=points_disc_ps, labels=input_labels_ps,points_cup=points_cup_ps, labels_cup = labels_cup_ps, box_buffer=sam_buffer,
                        multimask=False, use_points = use_points, use_boxes = use_boxes, use_masks = use_masks, logits_dpl = None, 
                        # rel_buffer=0.05, 
                        # buffer_type = 1
                        rel_buffer=0.15, buffer_type = 0
                        )
    
        
        dice_sam_cup = dice_coefficient_numpy( cup_masks, target_numpy[:, 0, ...])
        dice_sam_disc = dice_coefficient_numpy( dice_masks, target_numpy[:, 1, ...])
        dice_sam_vs_dpl = dice_coefficient_numpy( cup_masks, masks[:,0, ...])
        
        # print("!!! DICE DPL PSEUDO - CUP", cup_dice)
        # print("!!! DICE SAM - CUP", dice_sam_cup)
        # print("!!! DICE SAM VS DPL", dice_sam_vs_dpl)
        
        # gt_vs_dpl_dice_pts.append(cup_dice[0])
        if cup_masks.sum() / (512 * 512) > 0.3:
            print("Appending 0, cup too large")
            sam_vs_dpl_dice_pts_other.append(0)
        else:
            sam_vs_dpl_dice_pts_other.append(dice_sam_vs_dpl[0])
        # sam_dice_pts.append(dice_sam_cup[0])
        # all_names.append(img_name[0])

        # if show_vis:
        #     show_sam_dpl(view_img, img_name, pseudo_boxes, target_numpy, masks, cup_masks ,points_cup_ps, labels_cup_ps, dice_sam_cup, cup_dice)
        #     plt.show()
        
        
        dice_masks, cup_masks, scores, logits = get_sam_on_boxes(view_img, predictor, pseudo_boxes, points=points_disc_ps, labels=input_labels_ps,points_cup=points_cup_ps, labels_cup = labels_cup_ps, box_buffer=sam_buffer,
                                multimask=False, use_points = use_points, use_boxes = use_boxes, use_masks = use_masks, logits_dpl = None, 
                                # rel_buffer=0.075, 
                                # buffer_type = 0
                                 rel_buffer=0.075, buffer_type = 0
                                )
        
        
        dice_sam_cup = dice_coefficient_numpy( cup_masks, target_numpy[:, 0, ...])
        dice_sam_disc = dice_coefficient_numpy( dice_masks, target_numpy[:, 1, ...])
        dice_sam_vs_dpl = dice_coefficient_numpy( cup_masks, masks[:,0, ...])
        
        # print("!!! DICE DPL PSEUDO - CUP", cup_dice)
        # print("!!! DICE SAM - CUP", dice_sam_cup)
        # print("!!! DICE SAM VS DPL", dice_sam_vs_dpl)
        
        # gt_vs_dpl_dice_pts.append(cup_dice[0])
        if cup_masks.sum() / (512 * 512) > 0.3:
            print("Appending 0, cup too large")
            sam_vs_dpl_dice_pts_third.append(0)
        else:
            sam_vs_dpl_dice_pts_third.append(dice_sam_vs_dpl[0])
    
    # for nam
    # print("DICE SAM - DISC", dice_sam_disc)    
    threshold = 0.9
    higher_than_threshold = True
    save_pkl = True

    def image_predicate(val, threshold, higher_than_threshold):
        if higher_than_threshold is True:
            return val > threshold
        else:
            return val <= threshold
    
    ensembled_scores = []
    all_sam_vs_dpl_pts_list = [(sam_vs_dpl_dice_pts[i] + sam_vs_dpl_dice_pts_other[i] + sam_vs_dpl_dice_pts_third[i])/3 for i in range(len(sam_vs_dpl_dice_pts))]    
    zipped_array = list(zip(sam_vs_dpl_dice_pts_third, gt_vs_dpl_dice_pts, all_names))
    for (sam_score, _, img_name) in zipped_array:
        if not img_name in current_ranking:
            current_ranking[img_name] = sam_score
        else:
            current_ranking[img_name] = current_ranking[img_name]*0.5 + sam_score*0.5
        ensembled_scores.append(current_ranking[img_name])

    zipped_array = list(zip(ensembled_scores, gt_vs_dpl_dice_pts, all_names))
        
    for i,sam_vs_dpl_dice_pts in enumerate([sam_vs_dpl_dice_pts, sam_vs_dpl_dice_pts_other, sam_vs_dpl_dice_pts_third, all_sam_vs_dpl_pts_list, ensembled_scores]):
        
        # print(gt_vs_dpl_dice_pts)
        # print(sam_vs_dpl_dice_pts)


        coff = np.corrcoef(gt_vs_dpl_dice_pts, sam_vs_dpl_dice_pts)
        print("########## CORRELATION COEFFICIENT", coff[0][1])

        # print("SAM VS DPL MAX", max(sam_vs_dpl_dice_pts))
        # print("GT VS DPL MAX", max(gt_vs_dpl_dice_pts))
        # print("SAM VS DPL MIN", min(sam_vs_dpl_dice_pts))
        # print("GT VS DPL MIN", min(gt_vs_dpl_dice_pts))

        # thresholded_dpl = [gt_vs_dpl_dice_pts[i] for i in range(len(gt_vs_dpl_dice_pts)) if image_predicate(sam_vs_dpl_dice_pts[i], threshold, higher_than_threshold)]
        # thresholded_names = [all_names[i] for i in range(len(gt_vs_dpl_dice_pts)) if image_predicate(sam_vs_dpl_dice_pts[i], threshold, higher_than_threshold)]
        # print(len(thresholded_dpl))
        # print(len(gt_vs_dpl_dice_pts))

        # print("Avg of DPL dice after thresholding", sum(thresholded_dpl)/len(thresholded_dpl))
        print("Avg of DPL dice", sum(gt_vs_dpl_dice_pts)/len(gt_vs_dpl_dice_pts))
        print("Avg of SAM dice", sum(sam_dice_pts)/len(sam_dice_pts))

        # print("Thresholded names", thresholded_names)

        plt.figure(figsize=(8, 5))  # Adjust figure size if needed
        plt.axes()

        plt.ylabel("Dice - Pred with Ground Truth")
        plt.xlabel("Dice - Pred with SAM")
        
        plt.scatter(sam_vs_dpl_dice_pts,gt_vs_dpl_dice_pts, label='GT vs. DPL (y) Sam vs. DPL (x)')
        plt.title(f"Corr : {coff[0][1]}")
        plt.savefig(filename_pkl.replace(".pkl", f"_epoch_{epoch}_opt_{i}_cor_{round(coff[0][1], 4)*100}.png"))
        print("Saved plot to: ", filename_pkl.replace(".pkl", ".png"))

    def sort_key_by_sam(elem):
        return elem[0]
    sorted_zipped = sorted(zipped_array, key=sort_key_by_sam)
    if save_pkl:
        import pickle
        with open(filename_pkl, "wb") as f:
            print("Saving pkl to: ", filename_pkl)
            pickle.dump(sorted_zipped, f)



# 1. dataset 
# composed_transforms_test = transforms.Compose([
#     tr.Resize(512),
#     tr.Normalize_tf(),
#     tr.ToTensor()
# ])
# db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs', transform=composed_transforms_test)
# db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test/ROIs', transform=composed_transforms_test)
# db_source = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.source, split='train/ROIs', transform=composed_transforms_test)

# train_loader = DataLoader(db_train, batch_size=args.batchsize, shuffle=False, num_workers=1)
# test_loader = DataLoader(db_test, batch_size=args.batchsize, shuffle=False, num_workers=1)
# source_loader = DataLoader(db_source, batch_size=args.batchsize, shuffle=False, num_workers=1)

# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# model_file = args.model_file

# if args.dataset=="Domain2":        
#     npfilename = './results/prototype/pseudolabel_D2_0610_og.npz'
# elif args.dataset=="Domain1":        
#     npfilename = './results/prototype/pseudolabel_D1_0610_og.npz'

# print("loading pseudo labels from: ", npfilename)
# npdata = np.load(npfilename, allow_pickle=True)
# pseudo_label_dic = npdata['arr_0'].item()
# uncertain_dic = npdata['arr_1'].item()
# proto_pseudo_dic = npdata['arr_2'].item()

# 2. model
# model = DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
# model = netd_eval.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn).cuda()
# if torch.cuda.is_available():
#     model = model.cuda()
# print('==> Loading %s model file: %s' %
#       (model.__class__.__name__, model_file))
# checkpoint = torch.load(model_file)
# pretrained_dict = checkpoint['model_state_dict']
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
# model.load_state_dict(pretrained_dict)
# model.train()


# FOR RIMONE 
# PROMPT
# sam_points_cup = 6
# sam_radius_ratio_cup = 0.55
# better one: 8 0.65
# dirshti best: 8 0.8

# return generate_sam_score(get_denoised_mask, get_fixed_bbox, train_loader, predictor)
        