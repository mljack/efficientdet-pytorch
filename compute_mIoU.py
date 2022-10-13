import os
import sys
import json
import math
import pprint
import numpy as np
from shapely.geometry import box, Polygon
from matplotlib import pyplot as plt

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

class Obj:
    def __init__(self, obj_json):
        self.json = obj_json
        pts = obb_to_pts(obj_json)
        self.poly = Polygon(pts)
        self.radius = dist(pts[0], pts[2]) * 0.5
        self.p = ((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) * 0.25,
            (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) * 0.25)
        self.match_count = 0
            
def IoU_obb(obj1, obj2):
    dx = abs(obj1.p[0] - obj2.p[0])
    dy = abs(obj1.p[1] - obj2.p[1])
    radius = obj1.radius + obj2.radius
    if dx > radius or dy > radius:
        return 0.0
    intersection_area = obj1.poly.intersection(obj2.poly).area
    if intersection_area == 0.0:
        return 0.0
    return intersection_area / obj1.poly.union(obj2.poly).area

def compute_max_IoU_obj(obj_a, objs, func):
    max_IoU = 0.0
    max_IoU_obj = None
    for obj_b in objs:
        if obj_a is obj_b:
            continue
        iou = func(obj_a, obj_b)
        if iou > max_IoU:
            max_IoU = iou
            max_IoU_obj = obj_b
    #print(max_IoU)
    return max_IoU_obj, max_IoU

def load_vehicle_markers(path):
    with open(path) as f:
        j = json.load(f)
    return j

def obb_to_pts(obb):
    c = np.array((obb["x"], obb["y"]))
    yaw = math.radians(obb["heading_angle"])
    dir = np.array((math.cos(yaw), math.sin(yaw)))
    normal = np.array((-math.sin(yaw), math.cos(yaw)))
    length = obb["length"]
    width = obb["width"]
    return np.array([
        c - 0.5*length*dir - 0.5*width*normal,
        c - 0.5*length*dir + 0.5*width*normal,
        c + 0.5*length*dir + 0.5*width*normal,
        c + 0.5*length*dir - 0.5*width*normal,
    ])

def run(gt_label_path, obb_det_path, IoUs):
    gt_labels = load_vehicle_markers(gt_label_path)
    obb_dets = load_vehicle_markers(obb_det_path)
    gt_labels = [Obj(gt[0]) for gt in gt_labels]
    obb_dets = [Obj(obb[0]) for obb in obb_dets]
    gt_labels = [gt for gt in gt_labels if "enabled" not in gt.json or gt.json["enabled"]]
    obb_dets = [obb_det for obb_det in obb_dets if "enabled" not in obb_det.json or obb_det.json["enabled"]]

    matched_dets = []
    unmatched_dets = []
    missed_dets = []

    for obb_det in obb_dets:
        max_IoU_obj, max_IoU = compute_max_IoU_obj(obb_det, gt_labels, IoU_obb)
        if max_IoU > 0.1:
            max_IoU_obj.match_count += 1
            matched_dets.append(obb_det)
        else:
            unmatched_dets.append(obb_det)
        IoUs.append(max_IoU)
    for gt_label in gt_labels:
        if gt_label.match_count == 0:
            missed_dets.append(gt_label)

    for matched_det in matched_dets:
        matched_det.json["enabled"] = True
    for unmatched_det in unmatched_dets:
        unmatched_det.json["enabled"] = False
    for missed_det in missed_dets:
        missed_det.json["enabled"] = True
        missed_det.json["manually_created"] = True
        missed_det.json["certainty"] = 0.35
        #print(missed_det.json)
    all_objs = matched_dets + unmatched_dets + missed_dets
    for idx, obj in enumerate(all_objs):
        obj.json["id"] = idx
    all_objs = [[obj.json] for obj in all_objs]

    # with open(filtered_det_path, "w") as f:
    #     f.write(pprint.pformat(all_objs, width=300, indent=1).replace("'", "\"").replace("True", "true").replace("False", "false"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python compute_mIoU.py gt_label_path obb_det_path")
        exit(-1)
    gt_label_base_path = sys.argv[1]
    obb_det_base_path = sys.argv[2]

    IoUs = []
    for item in os.listdir(gt_label_base_path):
        if not item.endswith(".vehicle_markers.json"):# or item.find("002815.") != 0:
            continue
        gt_label_path = os.path.join(gt_label_base_path, item)
        obb_det_path = os.path.join(obb_det_base_path, item)
        if os.path.exists(obb_det_path):
            #print(obb_det_path)
            run(gt_label_path, obb_det_path, IoUs)

    print("matched[%s]:[%d], mIoU:[%.4f%%]" % (gt_label_base_path, len(IoUs), np.mean(IoUs)*100.0))
    #plt.hist(IoUs, 200)
    #plt.show()
