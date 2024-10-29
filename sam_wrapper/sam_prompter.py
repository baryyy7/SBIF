import numpy as np
from sam_wrapper import PointsInElipse

def get_points_from_boxes(boxes, n, n_cup = 0, radius_ratio=0.75, radius_ratio_cup = 0.75, buffer_ratio = 0.1, sam_neg=-1, inner_cup = True, inner_disc = True):
    inside_circle_ratio = radius_ratio
    size_cup = boxes[0][2] - boxes[0][0]
    size_disc = boxes[1][2] - boxes[1][0]
    size_cup_y = boxes[0][3] - boxes[0][1]
    size_disc_y = boxes[1][3] - boxes[1][1]
    
    cup_disc_x_diff = ((boxes[0][0] - boxes[1][0]) + (boxes[1][2] - boxes[0][2]))/2
    cup_disc_y_diff = ((boxes[0][1] - boxes[1][1]) + (boxes[1][3] - boxes[0][3]))/2
    buffer_x = cup_disc_x_diff * buffer_ratio
    buffer_y = cup_disc_y_diff * buffer_ratio
    
    cup_disc_x_diff = ((boxes[0][0] - boxes[1][0]) + (boxes[1][2] - boxes[0][2]))/2
    cup_disc_y_diff = ((boxes[0][1] - boxes[1][1]) + (boxes[1][3] - boxes[0][3]))/2
    
    # boxes[0][0] -= cup_disc_x_diff/2
    # boxes[0][2] += cup_disc_x_diff/2
    # boxes[0][1] -= cup_disc_y_diff/2
    # boxes[0][3] += cup_disc_y_diff/2
    
    points_cup = PointsInElipse(r2 = (size_cup_y * radius_ratio_cup) * 0.5 ,r1 = (size_cup * radius_ratio_cup) * 0.5 , n=n_cup)
    points_cup_inner = PointsInElipse(r2 = (size_cup_y * radius_ratio_cup) * 0.25 ,r1 = (size_cup * radius_ratio_cup) * 0.25 , n=n_cup//2)
    points_disc = PointsInElipse(r2 = (size_disc_y * inside_circle_ratio) * 0.5 ,r1 = (size_disc * inside_circle_ratio) * 0.5 , n=n)
    points_disc_inner = PointsInElipse(r2 = (size_disc_y * inside_circle_ratio) * 0.25 ,r1 = (size_disc * inside_circle_ratio) * 0.25 , n=n//2)
    # points_cup = PointsInCircum(r = (size_cup * radius_ratio_cup) * 0.5 , n=n)
    # points_cup_inner = PointsInCircum(r = (size_cup * radius_ratio_cup) * 0.25 , n=n)
    # points_disc = PointsInCircum(r = (size_disc * inside_circle_ratio) * 0.5 , n=n)
    
    middle_cup = (boxes[0][0] + size_cup / 2, boxes[0][1] + size_cup_y / 2)
    middle_disc = (boxes[1][0] + size_disc / 2, boxes[1][1] + size_disc_y / 2)
    
    
    points_cup  = np.array([(int(p[0] + middle_cup[0]) , int(p[1] + middle_cup[1])) for p in points_cup])    
    points_cup_inner  = np.array([(int(p[0] + middle_cup[0]) , int(p[1] + middle_cup[1])) for p in points_cup_inner])
    points_disc_inner  = np.array([(int(p[0] + middle_disc[0]) , int(p[1] + middle_disc[1])) for p in points_disc_inner])
    points_disc  = np.array([(int(p[0] + middle_disc[0]) , int(p[1] + middle_disc[1])) for p in points_disc])
    labels_cup = [1 for _ in points_cup]
    labels_disc = [1 for _ in points_disc]
    if inner_cup:
        labels_cup.extend([1 for _ in points_cup_inner])
        points_cup = np.concatenate((points_cup, points_cup_inner), axis=0)
    if inner_disc:
        labels_disc.extend([1 for _ in points_disc_inner])
        points_disc = np.concatenate((points_disc, points_disc_inner), axis=0)
    
    if sam_neg > 0:        
        # points_cup_neg = PointsInCircum(r = ((size_cup/2 + ((cup_disc_x_diff + cup_disc_y_diff) * 1/2)) ) * 0.465 , n=sam_neg)
        points_cup_neg = PointsInElipse(r2 = (0.5 * size_cup_y) + buffer_y , r1 = size_cup * 0.5 + buffer_x , n=sam_neg)
        # points_cup_neg.extend(PointsInCircum(r = ((size_cup + ((cup_disc_x_diff + cup_disc_y_diff)/2))) * 0.5 , n=sam_neg))
        points_cup_neg  = np.array([(int(p[0] + middle_cup[0]) , int(p[1] + middle_cup[1])) for p in points_cup_neg])
        
        points_cup_neg = np.array([(boxes[0][0], boxes[0][1]), (boxes[0][0], boxes[0][3]), (boxes[0][2], boxes[0][1]), (boxes[0][2], boxes[0][3])])
        
        if len(points_cup) > 0:
            all_cup = np.concatenate((points_cup, points_cup_neg), axis=0)
        else: 
            all_cup = points_cup_neg
        # print("ALLCUP SHAPEEEEE", all_cup.shape)
        # print("ALLdisc SHAPEEEEE", points_disc.shape)
        labels_cup_neg = [0 for _ in points_cup_neg]
        labels_cup.extend(labels_cup_neg)
    else: 
        all_cup = points_cup
    
    labels_cup = np.array(labels_cup)
    labels_disc = np.array([1 for _ in points_disc])
    return all_cup, points_disc, labels_cup, labels_disc

def get_sam_on_boxes(image, predictor, boxes, points=None, labels=None, points_cup=None, labels_cup = None, box_buffer = 0, multimask = False, use_points = True,
                     use_boxes = True, use_masks = False, logits_dpl = None, rel_buffer = 0, buffer_type = 0):
    
    # print(type(transformed_boxes))
    # print(points_cup.shape)
    # print(points.shape)
    # stacked_points = torch.tensor(np.stack([np.array(points_cup), np.array(points)]))
    # stacked_labels = torch.tensor(np.stack([np.array(labels_cup), np.array(labels)]))
    
    # transformed_boxes = predictor.transform.apply_boxes_torch(torch.tensor(boxes), image.shape[:2])
    # transformed_points = predictor.transform.apply_coords_torch(stacked_points)
    # transformed_labels = torch.tensor(labels).to(predictor.device)
    
    # masks, _, _ = predictor.predict_torch(
    #     point_coords=transformed_points,
    #     point_labels=stacked_labels,
    #     boxes=transformed_boxes,
    #     multimask_output=False,
    # )
    
    # cup_masks, dice_masks = masks[0], masks[1]
    from copy import deepcopy
    boxes = deepcopy(boxes)
    # print("buffer stats: " ,buffer_type, rel_buffer)
    if points_cup is not None:
        if buffer_type == 0:
            cup_disc_x_diff = ((boxes[0][0] - boxes[1][0]) + (boxes[1][2] - boxes[0][2]))/2
            cup_disc_y_diff = ((boxes[0][1] - boxes[1][1]) + (boxes[1][3] - boxes[0][3]))/2
            
            buffer_x = cup_disc_x_diff * rel_buffer
            buffer_y = cup_disc_y_diff * rel_buffer
        
        elif buffer_type ==1:
            cup_box_x_size = boxes[0][2] - boxes[0][0]
            cup_box_y_size = boxes[0][3] - boxes[0][1]
            
            buffer_x = cup_box_x_size * rel_buffer
            buffer_y = cup_box_y_size * rel_buffer
        # buffer_x = 0
        # buffer_y = 0
        
        # boxes[0][0] -= buffer_x
        # boxes[0][2] += buffer_x
        # boxes[0][1] -= buffer_y 
        # boxes[0][3] += buffer_y 
        
        # boxes[1][0] -= buffer_x
        # boxes[1][2] += buffer_x
        # boxes[1][1] -= buffer_y
        # boxes[1][3] += buffer_y
        
        boxes[0][0] -= buffer_x
        boxes[0][2] += buffer_x
        boxes[0][1] -= buffer_y
        boxes[0][3] += buffer_y
        
                
        # boxes[0][0] -= 0
        # boxes[0][2] += 0
        # boxes[0][1] -= 0
        # boxes[0][3] += 0
        
    
    
    dice_masks, scores_disc, logits_disc =   predictor.predict(
        point_coords=points if (len(points) > 0 and use_points) else None,
        point_labels=labels if (len(points) > 0 and use_points) else None,
        box=boxes[1].cpu().numpy() if use_boxes else None,
        multimask_output=multimask,
    )
    # print(dice_masks.shape)

    cup_masks, scores_cup, logits_cup = predictor.predict(
        point_coords=points_cup if (len(points_cup) > 0 and use_points) else None,
        point_labels=labels_cup if (len(points_cup) > 0 and use_points) else None,
        box=boxes[0].cpu().numpy() if use_boxes else None,
        multimask_output=multimask,
        mask_input = logits_dpl if use_masks else None
    )
    # print(cup_masks.shape)
    return dice_masks, cup_masks, scores_cup, logits_cup

class SamPrompter:
    def __init__(self, sam_points, sam_radius_ratio, boxes, use_points, use_boxes, fix_multi = 0.08, sam_fix_points = 7, sam_neg = 0 ):
        self.sam_points = sam_points
        self.sam_radius_ratio = sam_radius_ratio
        self.boxes = boxes
        self.use_points = use_points
        self.use_boxes = use_boxes
        self.fix_multi = fix_multi
        self.sam_fix_points = sam_fix_points
        self.sam_neg = sam_neg
        
class SamPrompterAll:
    def __init__(self,
                boxes,
                sam_points, sam_radius_ratio, 
                sam_points_cup, sam_radius_ratio_cup,
                use_points, use_boxes, initial_negative, fix_multi = 0.08, sam_fix_points = 7, sam_neg = 0, enable_inner_cup = False, enable_inner_disc = False):
        self._prompter_disc = SamPrompter(sam_points, sam_radius_ratio, boxes, use_points, use_boxes, fix_multi, sam_fix_points, sam_neg)
        self._prompter_cup = SamPrompter(sam_points_cup, sam_radius_ratio_cup, boxes, use_points, use_boxes, fix_multi, sam_fix_points, sam_neg)
        self.sam_neg = sam_neg
        self.enable_inner_cup = enable_inner_cup
        self.enable_inner_disc = enable_inner_disc
        self.boxes = boxes
        self.initial_negative = initial_negative
    
    
    def get_points_from_boxes_sam(self, radius_cup=None, num_cup=None):
        rad_cup = radius_cup if radius_cup is not None else self._prompter_cup.sam_radius_ratio
        num_cup_updated = num_cup if num_cup is not None else self._prompter_cup.sam_points
        return get_points_from_boxes(self.boxes, n=self._prompter_disc.sam_points, n_cup = num_cup_updated, radius_ratio=self._prompter_disc.sam_radius_ratio,
                              radius_ratio_cup=rad_cup ,sam_neg = self.sam_neg, inner_cup = self.enable_inner_cup, inner_disc = self.enable_inner_disc)
    
    def add_negative_on_bbox(self):
        return self.initial_negative