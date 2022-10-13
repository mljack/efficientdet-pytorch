import os
import sys
import json
import math
import pprint
import numpy as np
from shapely.geometry import box, Polygon

def dist(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

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
    max_IoU = 0.01
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

def run(aabb_label_path, obb_det_path, filtered_det_path):
    aabb_labels = load_vehicle_markers(aabb_label_path)
    obb_dets = load_vehicle_markers(obb_det_path)
    aabb_labels = [Obj(aabb[0]) for aabb in aabb_labels]
    obb_dets = [Obj(obb[0]) for obb in obb_dets]
    matched_dets = []
    unmatched_dets = []
    missed_dets = []
    for obb_det in obb_dets:
        max_IoU_obj, max_IoU = compute_max_IoU_obj(obb_det, aabb_labels, IoU_obb)
        if max_IoU > 0.1:
            max_IoU_obj.match_count += 1
            matched_dets.append(obb_det)
        else:
            unmatched_dets.append(obb_det)
    for aabb_label in aabb_labels:
        if aabb_label.match_count == 0:
            missed_dets.append(aabb_label)

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

    with open(filtered_det_path, "w") as f:
        f.write(pprint.pformat(all_objs, width=300, indent=1).replace("'", "\"").replace("True", "true").replace("False", "false"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python filter_obb_det_by_aabb_labels.py aabb_label_path obb_det_path")
        exit(-1)
    aabb_label_base_path = sys.argv[1]
    obb_det_base_path = sys.argv[2]
    filtered_det_base_path = obb_det_base_path.replace("_det", "_det_filtered")
    if not os.path.exists(filtered_det_base_path):
        os.mkdir(filtered_det_base_path)
    for item in os.listdir(aabb_label_base_path):
        if not item.endswith(".vehicle_markers.json"):# or item.find("002815.") != 0:
            continue
        aabb_label_path = os.path.join(aabb_label_base_path, item)
        obb_det_path = os.path.join(obb_det_base_path, item)
        filtered_det_path = os.path.join(filtered_det_base_path, item)
        if os.path.exists(obb_det_path):
            print(obb_det_path)
            run(aabb_label_path, obb_det_path, filtered_det_path)