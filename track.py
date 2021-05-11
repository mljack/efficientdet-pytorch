import time
import os
import sys
import math
import json
import collections
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from scipy.spatial.distance import cdist
#from scipy.optimize import linear_sum_assignment
from shapely.geometry import box, Polygon
import pprint
#import gc
#from pympler import muppy, summary

def dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def IoU(b1, b2):
    intersection = [max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])]
    if intersection[2] < intersection[0] or intersection[3] < intersection[1]:
        return 0.0
    intersection_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
    union_area = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - intersection_area
    return intersection_area / union_area

def IoU_poly_vs_refined_poly(obj1, obj2, debug=False):
    dx = abs(obj1.p[0] - obj2.p[0])
    dy = abs(obj1.p[1] - obj2.p[1])
    radius = obj1.radius + obj2.radius
    if dx > radius or dy > radius:
        if debug:
            print("A", dx, dy, obj1.radius, obj2.radius)
        return 0.0
    poly = obj2.poly if obj2.refined_poly is None else obj2.refined_poly
    intersection_area = obj1.poly.intersection(poly).area
    if intersection_area == 0.0:
        if debug:
            print("B")
        return 0.0
    if debug:
        print("C")
    return intersection_area / obj1.poly.union(poly).area

def IoU_refined_poly_vs_refined_poly(obj1, obj2, debug=False):
    dx = abs(obj1.p[0] - obj2.p[0])
    dy = abs(obj1.p[1] - obj2.p[1])
    radius = obj1.radius + obj2.radius
    if dx > radius or dy > radius:
        if debug:
            print("A", dx, dy, obj1.radius, obj2.radius)
        return 0.0
    intersection_area = obj1.refined_poly.intersection(obj2.refined_poly).area
    if intersection_area == 0.0:
        if debug:
            print("B")
        return 0.0
    if debug:
        print("C")
    return intersection_area / obj1.refined_poly.union(obj2.refined_poly).area

class TrackedObj:
    def __init__(self, obj):
        self.track_id = None
        self.next_objs = []
        self.next_obj_ious = []
        self.frames = collections.OrderedDict()
        self.inactive_count = 0
        self.refined_poly = None
        self.discarded = False
        self.angle_count = 1

        self.p = obj.p
        self.radius = obj.radius

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

class Obj:
    def __init__(self, id, obj_json):
        self.json = obj_json
        self.detection_id = id
        self.label = obj_json["label"]
        self.score = obj_json["score"]
        pts = obj_json["polygon"]
        self.box = pts
        self.poly = Polygon(pts)
        self.radius = dist(pts[0], pts[2]) * 0.5
        self.p = ((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) * 0.25,
            (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) * 0.25)

def max_IoU_obj(obj_a, objs, func):
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

def last_item(dict_a):
    if len(dict_a) == 0:
        return []
    return next(reversed(dict_a)).values()

def in_range(obj):
    return False
    #return obj.p[0] >= 2500 and obj.p[0] <= 2609+179 and obj.p[1] >= 1000 and obj.p[1] <= 1068+66
    #return obj.p[1] >= 1031 and obj.p[1] <= 1031+60 and obj.p[0] >= 3464 and obj.p[0] <= 3464+150
    #return obj.p[0] >= 1364 and obj.p[0] <= 1364+176 and obj.p[1] >= 1124 and obj.p[1] <= 1124+103

def debug_print_objs(active_tracked_objs, inactive_tracked_objs, angle):
    return

    print("active_tracked_objs:")
    for id, obj in active_tracked_objs.items():
        if in_range(obj):
            print("\t", id, list(obj.frames.keys()))
            for key, angle_objs in obj.frames.items():
                print("\t\t", key, list(angle_objs.keys()))
    print("inactive_tracked_objs:")
    for id, obj in inactive_tracked_objs.items():
        if in_range(obj):
            print("\t", id, list(obj.frames.keys()))
            for key, angle_objs in obj.frames.items():
                print("\t\t", key, list(angle_objs.keys()))
    print("angle: ", angle)

def debug_show_objs(json_frame_id, obj, img, active_tracked_objs, iou, angle):
    if not in_range(obj):
        return

    print("iou", iou, json_frame_id, angle)

    if json_frame_id >= 40:
    #if json_frame_id == 17 and angle == 0:
        c = (255, 255, 255)
        pts = np.int32(np.array([(x, y) for x,y in obj.poly.exterior.coords]))
        img = cv2.drawContours(img, [pts], 0, c, 1)

        for obj2 in active_tracked_objs.values():
            if in_range(obj2):
                c = (0, 0, 0)
                pts = np.int32(np.array([(x,y) for x,y in obj2.refined_poly.exterior.coords]))
                img = cv2.drawContours(img, [pts], 0, c, 1)
                print(obj.p, obj2.p)
                print(obj.box)
                cv2.circle(img,(int(obj.p[1]), int(obj.p[0])), 2, (0,255,0), -1)
                cv2.circle(img,(int(obj2.p[1]), int(obj2.p[0])), 2, (255,0,0), -1)
                print(IoU_poly_vs_refined_poly(obj, obj2, debug=True))

        #cv2.imshow("result", img)
        #img2 = img[1124:1124+103, 1364:1364+176]
        img2 = img[1000:1068+66, 2500:2609+179]

        img2 = cv2.resize(img2, (img2.shape[1]*4, img2.shape[0]*4))
        cv2.imshow("result", img2)
        #cv2.imwrite("saved.png", img)
        #exit(0)
        k = cv2.waitKey(0)
        if k == 27:
            exit(0)
            #break

def read_json_objs(base_path):
    files = sorted(os.listdir(base_path))
    json_frame_id_old = -1
    start = time.time()
    for item in files:
        if item.find(".json") == -1 or item.find(".tracked.json") != -1:
            continue
        with open(os.path.join(base_path, item)) as f:
            yield item, json.load(f)
    yield "99999_000.json", None

def rotation_mat(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return M, nW, nH

def poly2aabb(poly):
    x0 = np.min(poly[:,0])
    x1 = np.max(poly[:,0])
    y0 = np.min(poly[:,1])
    y1 = np.max(poly[:,1])
    return np.array([(x0,y0),(x1,y0),(x1,y1),[x0,y1]])

def poly2aabb2(poly):
    x0 = np.min(poly[:,0])
    x1 = np.max(poly[:,0])
    y0 = np.min(poly[:,1])
    y1 = np.max(poly[:,1])
    return np.array([x0,y0,x1,y1])

def compute_iou(image, ref_poly, ref_angle, poly, angle):
    M, nW, nH = rotation_mat(image, angle)
    poly2 = cv2.transform(poly, M)[0,:,:]
    ref_poly2 = cv2.transform(ref_poly, M)[0,:,:]
    iou = IoU(poly2aabb2(ref_poly2), poly2aabb2(poly2))
    #print("IoU: ", iou)

    if 0:
        image = image.copy()
        image = cv2.warpAffine(image, M, (nW, nH))
        image = cv2.drawContours(image, np.int32([poly2]), 0, (0, 255, 255), 1)
        image = cv2.drawContours(image, [np.int32(ref_poly2)], 0, (255, 0, 255), 1)
        image = cv2.drawContours(image, [np.int32(poly2aabb(ref_poly2))], 0, (255, 0, 0), 1)
        #cv2.imwrite("rotated.png", image*255)
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        #cv2.setWindowProperty("input", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("img", image)
        k = cv2.waitKey()
        if k == 27:
            exit(0)

    return iou

def compute_attrs(obj, img):
    boxes = obj["boxes"]
    if len(boxes) == 0:
        return {}
    if len(boxes) == 1:
        poly = boxes[0]["polygon"]
        obj["poly"] = poly
        x = (poly[0][0]+poly[1][0]+poly[2][0]+poly[3][0]) * 0.25
        y = (poly[0][1]+poly[1][1]+poly[2][1]+poly[3][1]) * 0.25
        obj["center"] = [round(x, 4), round(y, 4)]
        obj["angle"] = 9999.0
        obj["length"] = 9999.0
        obj["width"] = 9999.0
        obj["score"] = boxes[0]["score"]
        obj["certainty"] = 0.0
        obj["certainty2"] = 0.0
        obj["box_count"] = 1
    else:
        min_area2 = 1e30
        min_area_angle = None
        min_area_length2 = None
        min_area_width2 = None
        min_area_poly = None
        min_area_score = None
        min_area_idx = -1
        aabb = None
        aabb_score = None
        for idx, box in enumerate(boxes):
            poly = box["polygon"]
            if box["angle"] == 0.0:
                aabb = poly
                aabb_score = box["score"]
            angle = math.degrees(math.atan2(poly[2][1]-poly[1][1], poly[2][0]-poly[1][0]))
            width2 = dist2(poly[0], poly[1])
            length2 = dist2(poly[1], poly[2])
            area2 = length2 * width2
            if area2 < min_area2:
                min_area_idx = idx
                min_area2 = area2
                min_area_angle = angle
                min_area_length2 = length2
                min_area_width2 = width2
                min_area_poly = poly
                min_area_score = box["score"]
        if min_area_length2 < min_area_width2:
            min_area_width2, min_area_length2 = (min_area_length2, min_area_width2)
            min_area_angle += 90.0
        poly = min_area_poly
        x = (poly[0][0]+poly[1][0]+poly[2][0]+poly[3][0]) * 0.25
        y = (poly[0][1]+poly[1][1]+poly[2][1]+poly[3][1]) * 0.25
        poly = min_area_poly if aabb is None else aabb
        obj["score"] = min_area_score if aabb_score is None else aabb_score
        obj["poly"] = min_area_poly
        obj["center"] = [round(x, 4), round(y, 4)]
        obj["angle"] = min_area_angle
        obj["length"] = round(math.sqrt(min_area_length2), 4)
        obj["width"] = round(math.sqrt(min_area_width2), 4)
        obj["box_count"] = len(boxes)

        min_area_poly = np.array([min_area_poly])
        certainty_acc = 0.0
        certainty_acc2 = 0.0
        for idx, box in enumerate(boxes):
            if idx == min_area_idx:
                continue
            iou = compute_iou(img, min_area_poly, min_area_angle, np.array([box["polygon"]]), box['angle'])
            certainty_acc += iou
            certainty_acc2 += iou * box["score"]
        certainty = certainty_acc / len(range(0,90,5))
        certainty2 = certainty_acc2 / len(range(0,90,5))
        obj["certainty"] = certainty
        obj["certainty2"] = certainty2
        #print("[%3d][%3d boxes]: %.4f %.4f" % (obj['obj_id'], len(box), certainty, certainty2))

    min_x = min(poly[0][0], poly[1][0], poly[2][0], poly[3][0])
    min_y = min(poly[0][1], poly[1][1], poly[2][1], poly[3][1])
    max_x = max(poly[0][0], poly[1][0], poly[2][0], poly[3][0])
    max_y = max(poly[0][1], poly[1][1], poly[2][1], poly[3][1])
    obj["aabb"] = [[min_x, min_y], [max_x, max_y]]
    obj["label"] = 1.0

    del obj["boxes"]


def track(json_objs, img2, output_path=None, video_path=None, single_frame_obb=False):
    track_count = 0

    if video_path is not None:
        color = ((255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(0,0,0),(255,255,255),
                (180,0,0),(0,180,0),(0,0,180),(180,180,0),(180,0,180),(255,180,255),(255,255,180),
                (180,255,0),(180,0,255),(0,180,255),(255,180,0),(255,0,180),(0,255,180),(180,255,255))
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Resolution:\t%d x %d" % (frame_w, frame_h))
        print("FPS:\t\t%6.2f" % fps)
        print("Frame Count:\t", int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

        out_video_path = video_path[0:video_path.rfind(".")] + "_tracked.mp4"
        if sys.platform == "win32":
            fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_w,frame_h))
        print(out_video_path)

    active_tracked_objs = collections.OrderedDict()
    inactive_tracked_objs = collections.OrderedDict()
    to_delete = []
    objs = []
    new_tracks = []
    frame_json = []
    img = None
    frame_id = -1
    json_frame_id_old = -1
    start = time.time()
    for item, objs_json in json_objs:
        tokens = item.replace(".json", "").split("_")
        json_frame_id = int(tokens[0])
        angle = int(tokens[1]) if len(tokens) > 1 else 0

        #print(json_frame_id, angle)
        #if json_frame_id % 20 == 0 and angle not in {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85}:
        #    continue
        #elif json_frame_id % 20 != 0 and angle not in {0}:
        #    continue
        #if angle not in {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85}:
        #if angle not in {0, 10, 20, 30, 40, 50, 60, 70, 80}:
        #if angle not in {0, 15, 30, 15, 60, 75}:
        #if angle not in {0, 30, 60}:
        #if angle not in {0}:
        #    continue

        debug_print_objs(active_tracked_objs, inactive_tracked_objs, angle)
        
        is_new_frame = json_frame_id_old != -1 and json_frame_id_old != json_frame_id
        if is_new_frame:
            print("[%05d]: %.3fs" % (json_frame_id_old, time.time() - start))
            start = time.time()
            discarded_track_ids = set()

            if 1:
                # Mark objs with refined polygon overlapped as discarded
                max_iou3 = 0.0
                for idx_a, obj_a in enumerate(active_tracked_objs.values()):
                    if obj_a.track_id in discarded_track_ids:
                        continue
                    for idx_b, obj_b in enumerate(active_tracked_objs.values()):
                        if idx_a >= idx_b or obj_b.track_id in discarded_track_ids:
                            continue
                        iou3 = IoU_refined_poly_vs_refined_poly(obj_a, obj_b)
                        max_iou3 = max(max_iou3, iou3)
                        if iou3 > 0.2:
                            if len(obj_a.frames) > len(obj_b.frames):   # with less tracked frames
                                discarded_track_ids.add(obj_b.track_id)
                            elif len(obj_a.frames) < len(obj_b.frames):
                                discarded_track_ids.add(obj_a.track_id)
                            elif obj_a.refined_poly.area < obj_b.refined_poly.area: # with larger refined polygon
                                discarded_track_ids.add(obj_b.track_id)
                            else:
                                discarded_track_ids.add(obj_a.track_id)

            # Save to json
            if output_path is not None or single_frame_obb:
                for obj in active_tracked_objs.values():
                    if json_frame_id_old not in obj.frames:
                        continue
                    obj_json = {}
                    obj_json["obj_id"] = obj.track_id
                    if obj.track_id in discarded_track_ids:
                        obj_json["discarded"] = 1
                    #obj_json["refined_polygon"] = [[round(x,4), round(y,4)] for x,y in obj.refined_poly.exterior.coords]
                    obj_json["boxes"] = [{"label":angle_obj.label, "angle":angle, "score":angle_obj.score,
                        "polygon":[[round(p[0],4), round(p[1],4)]for p in angle_obj.box]}
                        for angle, angle_obj in obj.frames[json_frame_id_old].items()]   # boxes in all kinds of angles
                    compute_attrs(obj_json, img2)
                    frame_json.append(obj_json)
                if output_path is not None:
                    json_path = os.path.join(output_path, "%05d.tracked.json" % json_frame_id_old)
                    with open(json_path, 'w') as f:
                        f.write(pprint.pformat(frame_json, width=400, indent=1).replace("'", "\""))
                        #json.dump(frame_json, f)
                if single_frame_obb:
                    return frame_json
                frame_json.clear()

            # Move discarded tracks to inactive list
            for track_id in discarded_track_ids:
                obj = active_tracked_objs[track_id]
                obj.discarded = True
                #inactive_tracked_objs[track_id] = obj  # reduce memory consumption
                del active_tracked_objs[track_id]

            #gc.collect()
            #all_objects = muppy.get_objects()
            #sum1 = summary.summarize(all_objects)
            #summary.print_(sum1)

            # Show results
            if video_path is not None:
                for obj in active_tracked_objs.values():
                    pts = np.int32(np.array([(x,y) for x,y in obj.refined_poly.exterior.coords]))
                    img = cv2.drawContours(img, (pts,), 0, color[obj.track_id % len(color)], 3)
                out_video.write(img)

                cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("result",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.imshow("result", img)
                #cv2.imwrite("saved.png", img)
                k = cv2.waitKey(1)
                if k == 27:
                    break

        if objs_json is None: #  Done
            break

        if video_path is not None:
            while json_frame_id != frame_id or video_img is None:
                _, video_img = video.read()
                frame_id += 1
            img = video_img.copy()
         
        for idx, obj_json in enumerate(objs_json):
            objs.append(Obj(idx, obj_json))
        objs_json.clear()

        # Track with IoU
        for obj in objs:
            matched_obj, iou = max_IoU_obj(obj, active_tracked_objs.values(), IoU_poly_vs_refined_poly)
            debug_show_objs(json_frame_id, obj, img, active_tracked_objs, iou, angle)

            if matched_obj is None:
                new_tracks.append(obj)
            else:
                obj.track_id = matched_obj.track_id
                matched_obj.next_objs.append(obj)
                matched_obj.next_obj_ious.append(iou)
        objs.clear()

        # Split track that has multiple next objs in the same frame
        for obj in active_tracked_objs.values():
            if len(obj.next_objs) == 0:
                continue
            max_iou = max(obj.next_obj_ious)
            kk = -1
            for k, next_obj in enumerate(obj.next_objs):
                if obj.next_obj_ious[k] == max_iou:
                    selected_obj = next_obj
                    kk = k
            if 1:
                # Ignore highly overlapped detections
                for k, next_obj in enumerate(obj.next_objs):
                    if k != kk:
                        iou2 = selected_obj.poly.intersection(next_obj.poly).area / selected_obj.poly.union(next_obj.poly).area
                        if iou2 < 0.9:
                            new_tracks.append(next_obj)
            obj.next_objs.clear()
            obj.next_objs.append(selected_obj)
            obj.next_obj_ious.clear()
            obj.next_obj_ious.append(max_iou)

        # Add new tracks for non-matched objs
        for obj in new_tracks:
            tracked_obj = TrackedObj(obj)
            tracked_obj.track_id = track_count
            tracked_obj.refined_poly = copy.deepcopy(obj.poly)
            tracked_obj.frames[json_frame_id] = collections.OrderedDict()
            tracked_obj.frames[json_frame_id][angle] = obj
            active_tracked_objs[track_count] = tracked_obj
            track_count += 1
        new_tracks.clear()
        
        # Store matched results to frames
        to_delete.clear()
        for obj in active_tracked_objs.values():
            # Handle non-matched objects from previous frame
            if is_new_frame:
                if json_frame_id_old not in obj.frames: # no match for any angles
                    obj.inactive_count += 1
                    if obj.inactive_count >= 8:
                        to_delete.append(obj.track_id)
                    continue

            if len(obj.next_objs) > 0:
                obj.inactive_count = 0
                if json_frame_id not in obj.frames.keys():  # new frame
                    obj.frames.clear()  # reduce memory consumption
                    obj.frames[json_frame_id] = collections.OrderedDict()
                    obj.refined_poly = copy.deepcopy(obj.next_objs[0].poly)
                    obj.p = obj.next_objs[0].p
                    obj.angle_count = 1
                else:   # different angles
                    obj.refined_poly = obj.refined_poly.intersection(obj.next_objs[0].poly)
                    obj.angle_count += 1
                obj.frames[json_frame_id][angle] = obj.next_objs[0]
                obj.next_objs = []
                obj.next_obj_ious = []

        # Remove objs that has been inactive for a while
        for track_id in to_delete:
            #inactive_tracked_objs[track_id] = active_tracked_objs[track_id]    # reduce memory consumption
            del active_tracked_objs[track_id]

        json_frame_id_old = json_frame_id

    if video_path is not None:
        out_video.release()
        video.release()
        print(len(active_tracked_objs) + len(inactive_tracked_objs))

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python track.py detection-folder [video.mp4]")
    else:
        video_path = sys.argv[2] if len(sys.argv) == 3 else None
        track(read_json_objs(sys.argv[1]), output_path=sys.argv[1], video_path=video_path)