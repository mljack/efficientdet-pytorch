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

def run(dst_folder, filename, base_paths):
    labels = []
    for model_idx, base_path in enumerate(base_paths):
        path = os.path.join(base_path, filename)
        objs = load_vehicle_markers(path)
        for obj in objs:
            obj[0]["model_id"] = model_idx
        labels += objs

    for idx, obj in enumerate(labels):
        obj[0]["id"] = idx

    with open(os.path.join(dst_folder, filename), "w") as f:
        f.write(pprint.pformat(labels, width=500, indent=1).replace("'", "\"").replace("True", "true").replace("False", "false"))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("python concat_vehicle_markers.py dst_folder folder1 folder2 ...")
        exit(-1)
    dst_folder = sys.argv[1]
    base_paths = sys.argv[2:]
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    filenames = [item for item in os.listdir(base_paths[0]) if item.endswith(".vehicle_markers.json")]
    for filename in filenames:
        run(dst_folder, filename, base_paths)

