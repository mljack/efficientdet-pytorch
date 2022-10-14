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
        self.p = ((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) * 0.25,
            (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) * 0.25)
        self.poly = Polygon(pts)
        pts = pts_to_aabb_pts(pts)
        self.poly_aabb = Polygon(pts)
        self.radius = dist(pts[0], pts[2]) * 0.5
        self.match_count = 0
        self.dets = []
        self.IoU = 0.0

def IoU_obb(obj1, obj2):
    dx = abs(obj1.p[0] - obj2.p[0])
    dy = abs(obj1.p[1] - obj2.p[1])
    radius = obj1.radius + obj2.radius
    if dx > radius or dy > radius:
        return 0.0
    intersection_area = obj1.poly_aabb.intersection(obj2.poly_aabb).area
    if intersection_area == 0.0:
        return 0.0
    return intersection_area / obj1.poly_aabb.union(obj2.poly_aabb).area

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

def find_most_prob_obb(dets):
    max_IoU = 0.0
    max_IoU_det = None
    for det in dets:
        if det.IoU > max_IoU:
            max_IoU = det.IoU
            max_IoU_det = det
    return max_IoU_det

def load_vehicle_markers(path):
    with open(path) as f:
        j = json.load(f)
    return [obj[0] for obj in j if "enabled" not in obj[0] or obj[0]["enabled"]]

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

def pts_to_aabb_pts(pts):
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    return np.array([
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_max),
        (x_max, y_min),
    ])

def fuse_dets(dets):
    result = None
    for det in dets:
        certainty = det.json["certainty"] if "certainty" in det.json else -1.0
        score = det.json["score"]
        if (certainty > 0.75 and score > 0.5) or (certainty == -1 and score > 0.7):
            if result is None:
                result = det.poly
            else:
                result = result.intersection(det.poly)
    return result

def build_vehicle_marker(id, poly):
    obb = poly.minimum_rotated_rectangle
    pts = np.array([(x, y) for idx, (x, y) in enumerate(zip(obb.exterior.xy[0], obb.exterior.xy[1])) if idx < 4])
    c = np.mean(pts, axis=0)
    length = np.linalg.norm(pts[0] - pts[1])
    width = np.linalg.norm(pts[1] - pts[2])
    delta = pts[1] - pts[0]
    yaw = math.degrees(math.atan2(delta[1], delta[0]))
    if length < width:
        length, width = width, length
        yaw = (yaw + 90.0) % 360.0

    return [{
        "frame_id": 0,
        "id": id,
        "heading_angle": round(yaw, 1),
        "width": width,
        "length": length,
        "x": c[0],
        "y": c[1],
        "manually_keyed": False,
        "score": 1.0,
        "certainty": 1.0,
        "certainty2": 1.0,
    }]

def run(dst_folder, filename, base_paths):
    dst_json_path = os.path.join(dst_folder, filename)
    # if os.path.exists(dst_json_path):
    #     return
    print(dst_json_path)

    labels = []
    gt_objs = None
    for model_idx, base_path in enumerate(base_paths):
        path = os.path.join(base_path, filename)
        objs = load_vehicle_markers(path)
        objs = [Obj(obj) for obj in objs]
        if gt_objs is None:
            gt_objs = objs
        else:
            # Test AABB IoU with AABB groundtruth labels
            for gt_obj in gt_objs:
                max_IoU_obj, max_IoU = compute_max_IoU_obj(gt_obj, objs, IoU_obb)
                if max_IoU > 0.3:
                    max_IoU_obj.IoU = max_IoU
                    gt_obj.dets.append(max_IoU_obj)

    new_gt_objs = []
    count = -1
    for gt_obj in gt_objs:
        poly = fuse_dets(gt_obj.dets)
        count += 1
        if poly is not None and not poly.is_empty:
            # Intersection of all confident detection results
            new_gt_objs.append(build_vehicle_marker(count, poly))   # in red
        else:
            # Fallback to detection results with max IoU
            det = find_most_prob_obb(gt_obj.dets)
            if det is not None:
                det_json = det.json.copy()
                det_json["id"] = count
                det_json["certainty"] = 1.0
                det_json["score"] = 0.6             # in green
                new_gt_objs.append([det_json])
            else:
                # Fallback to AABB groundtruth
                gt_obj.json["id"] = count
                gt_obj.json["certainty"] = 0.5      # highlighted
                new_gt_objs.append([gt_obj.json])
        # Add all matched result in the hidden state.
        for obj_det in gt_obj.dets:
            count += 1
            obj_det.json["id"] = count
            obj_det.json["enabled"] = False
            new_gt_objs.append([obj_det.json])
    
    with open(dst_json_path, "w") as f:
        f.write(pprint.pformat(new_gt_objs, width=500, indent=1).replace("'", "\"").replace("True", "true").replace("False", "false"))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("python fuse_obb.py dst_folder folder1 folder2 ...")
        exit(-1)
    dst_folder = sys.argv[1]
    base_paths = sys.argv[2:]
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    filenames = [item for item in os.listdir(base_paths[0]) if item.endswith(".vehicle_markers.json")]
    for filename in filenames:
        #if filename == "003334.vehicle_markers.json":
        run(dst_folder, filename, base_paths)


#https://stackoverflow.com/questions/14697442/faster-way-of-polygon-intersection-with-shapely