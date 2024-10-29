import math
pi = math.pi  
import numpy as np
from skimage.measure import label, regionprops
from skimage import measure




def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def PointsInCircum(r,n=100):
    if n == 0:
        return []
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]

def PointsInElipse(r1, r2, n=100):
    
    ellipsePoints = [
    (r1 * math.cos(theta), r2 * math.sin(theta))
    for theta in (math.pi*2 * i/n for i in range(n))

    ]
    return ellipsePoints

def CreateMaskNumpyFromRadiusAndCenter(center, radius, shape):
    mask = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (i - center[0])**2 + (j - center[1])**2 < radius**2:
                mask[i, j] = 1
    return mask

def GetClosestPoint(points, point, num = 1, fix_multi=1, max_distance=100000):
    dis_array = np.sqrt(np.sum((points-point)**2, axis=1, keepdims=True))
    # .argmin(axis=0)
    print("shape of array", dis_array.shape)
    rel_idx = np.nonzero(dis_array < max_distance, )
    num_relevant = len(rel_idx[0])
    print(f"num_relevant: {num_relevant}")
    # print(f"{rel_idx}")
    
    relevant_skips = int(num_relevant / num )
    print(f"relevant_skips: {relevant_skips}")
    
    # num_multiplied = int(num // fix_multi)
    # print(num_multiplied)
    # skips = 1 // fix_multi
    
    indices = np.argpartition(dis_array, num_relevant, axis=0)[:num_relevant]
    print(max_distance)
    chosen_indices = [p for p in indices if p%relevant_skips == 0 and dis_array[p] < max_distance]
    print("Returning {} points".format(len(chosen_indices)))
    return chosen_indices

# def GetClosestPoint(points, point):
#     min_dist = 100000000
#     closest_point = None
#     for p in points:
#         dist = np.linalg.norm(np.array(p) - np.array(point))
#         if dist < min_dist:
#             min_dist = dist
#             closest_point = p
#     return closest_point

def get_point_index(points, ratio):
    return int(len(points) * ratio)
    
def get_neg_points_on_boxes(boxes, masks, ratio=0.2):
    # print("Start get neg!")
    # return np.array([
    #     [boxes[0][0], boxes[0][0]],
    #     [boxes[0][0], boxes[0][3]],
    #     [boxes[0][2], boxes[0][0]],
    #     [boxes[0][2], boxes[0][3]],
        
    # ])
    box = boxes[0].cpu().numpy().astype(np.uint16)
    masks = masks.cpu().numpy()
    boundary_mask = np.zeros_like(masks[0])
    # print(box[0])
    for i in range(box[0], box[2] + 1):
        for j in range(box[1], box[3] + 1):
            if i == box[0] or i == box[2] or j == box[1] or j == box[3]:
                boundary_mask[i, j] = 1
    cand_mask = boundary_mask * (1 - masks[0])
    
    # print("Create cand mask")
    
    size_cup = boxes[0][2] - boxes[0][0]
    # size_disc = boxes[1][2] - boxes[1][0]
    size_cup_y = boxes[0][3] - boxes[0][1]
    # size_disc_y = boxes[1][3] - boxes[1][1]
    middle_cup = (int(boxes[0][0] + size_cup / 2), int(boxes[0][1] + size_cup_y / 2))
    # middle_disc = (int(boxes[1][0] + size_disc / 2), int(boxes[1][1] + size_disc_y / 2))
    
    # print("Get middle")
    # print(middle_cup)
    # print(middle_cup[0])
    
    # get point to the right of middle cup and true on cand_mask
    cand_points = cand_mask.nonzero()
    r_points = np.where((cand_mask.nonzero()[0] > middle_cup[0]) * (cand_mask.nonzero()[1] > middle_cup[1]))
    l_points = np.where((cand_mask.nonzero()[0] < middle_cup[0]) * (cand_mask.nonzero()[1] > middle_cup[1]))
    t_points = np.where((cand_mask.nonzero()[1] < middle_cup[1]) * (cand_mask.nonzero()[0] < middle_cup[0]))
    b_points = np.where((cand_mask.nonzero()[1] < middle_cup[1]) * (cand_mask.nonzero()[0] > middle_cup[0]))
    
    r_idx = get_point_index(r_points[0], ratio)
    l_idx = get_point_index(l_points[0], ratio)
    t_idx = get_point_index(t_points[0], ratio)
    b_idx = get_point_index(b_points[0], ratio)
    
    p_list = []
    all_p = [r_points, l_points, t_points, b_points]
    all_idx = [r_idx, l_idx, t_idx, b_idx]
    for points, idx in zip(all_p, all_idx):
        try:
            p_list.append([cand_points[0][points[0][idx]], cand_points[1][points[0][idx]]])
        except BaseException as e:
            print(e)
    
    points = np.array(p_list)

    # print(points)
    neg_labels_cup = np.array([0 for _ in points])
    return points, neg_labels_cup
    
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

def enlarge_boxes(boxes):
    boxes[0][0] = int(boxes[0][0] * 0.9)
    boxes[0][1] = int(boxes[0][1] * 0.9)
    boxes[0][2] = int(boxes[0][0] * 1.1)
    boxes[0][3] = int(boxes[0][1] * 1.1)
    
    boxes[1][0] = int(boxes[0][0] * 0.7)
    boxes[1][1] = int(boxes[0][1] * 0.7)
    boxes[1][2] = int(boxes[0][0] * 1.3)
    boxes[1][3] = int(boxes[0][1] * 1.3)
    
    return boxes

def extract_from_bbox(max_props, is_pos = True, mode="mid", bbox_list = None, num_pos = 0):
    minr, minc, maxr, maxc = max_props.bbox
    bbox_fixed = (minc,minr,maxc,maxr)
    mid_bbox = [(minc + maxc) / 2, (minr + maxr) / 2]
    
    def sort_coords_key(coor):
        return coor[1]
    def sort_coords_key_x(coor):
        return coor[0]
    
    sorted_coords_by_y = sorted(max_props.coords, key = sort_coords_key)
    # print("mid y point", sorted_coords_by_y[len(sorted_coords_by_y) // 2])
    point_candidates = [c for c in sorted_coords_by_y if c[1] == sorted_coords_by_y[len(sorted_coords_by_y) // 2][1]]
    # print("point cands ", point_candidates)
    sorted_coords = sorted(point_candidates, key = sort_coords_key_x)
    mid_coor = sorted_coords_by_y[len(sorted_coords_by_y) // 2]
    # print("mid coor", mid_coor)
    p_list = []
    if mode == "mid":
        p_list = [mid_coor[1], mid_coor[0]]
    
    
    added_points = np.array([
        # mid_bbox,
        # [max_props.centroid[1],max_props.centroid[0]],
        [mid_coor[1], mid_coor[0]]
    ])
    label = 1 if is_pos else 0
    ret_labels = np.array([label for _ in range(added_points.shape[0])])
    return added_points, ret_labels, bbox_fixed, bbox_list, num_pos

def get_improvement_point_adaptive(prediction, cup_masks, mode="mid"):
    minus_labels = prediction[0] * (1 - cup_masks[0])
    max_props = None
    max_box = 0
    is_pos = True
    num_pos = 0
    
    print("minus labels sum", minus_labels.sum())
    
    a = measure.label(minus_labels)
    bbox_list_pos = []
    for curim in regionprops(a):        
        if max_box < curim.area:
            max_box = curim.area
            max_props = curim
        minr, minc, maxr, maxc = curim.bbox
        bbox_fixed = (minc,minr,maxc,maxr)
        bbox_list_pos.append(bbox_fixed)
    
    bbox_list_neg = []
    
    minus_labels = (1 - prediction[0]) * cup_masks[0]
    a = measure.label(minus_labels)
    for curim in regionprops(a):        
        if (is_pos and 2 * max_box < curim.area) or\
        (not is_pos and max_box < curim.area):
            max_box = curim.area
            max_props = curim
            is_pos = False
        minr, minc, maxr, maxc = curim.bbox
        bbox_fixed = (minc,minr,maxc,maxr)
        bbox_list_neg.append(bbox_fixed)
    
    num_pos = len(bbox_list_pos)
    if is_pos:
        bbox_list = bbox_list_pos
        bbox_list.extend(bbox_list_neg)
    else:
        bbox_list = bbox_list_pos
        bbox_list.extend(bbox_list_neg)
    
    if is_pos:
        print("Adding positive point")
    else:
        print("Adding negative point")
    return extract_from_bbox(max_props, is_pos=is_pos, mode=mode, bbox_list=bbox_list, num_pos = num_pos )
    

def get_improvement_point(prediction, cup_masks, neg = False, mode="mid"):
    minus_labels = prediction[0][0] * (1 - cup_masks[0])
    if neg:
        minus_labels = (1 - prediction[0][0]) * cup_masks[0]
    # print("HEYYY SHPAE", minus_labels.shape)
    a = measure.label(minus_labels)
    max_props = None
    max_box = 0
    # print(prediction.shape)
    # print(a.shape)
    bbox_list = []
    for curim in regionprops(a):
        print("Props")
        curim.area_bbox
        if max_box < curim.area_bbox:
            max_box = curim.area_bbox
            max_props = curim
        minr, minc, maxr, maxc = curim.bbox
        bbox_fixed = (minc,minr,maxc,maxr)
        bbox_list.append(bbox_fixed)
        # minr, minc, maxr, maxc = curim.bbox
        # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
        #                          fill=False, edgecolor=colors[i], linewidth=2)
        # plt.gca().add_patch(rect)
        
    return extract_from_bbox(max_props, is_pos=not neg, mode=mode, bbox_list = bbox_list)

