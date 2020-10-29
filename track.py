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

class Obj:
    def __init__(self, id, obj_json):
        self.json = obj_json
        box = obj_json["box"]
        self.detection_id = id
        self.box = box
        if "polygon" in obj_json.keys():
            pts = obj_json["polygon"]
        else:
            pts = [[box[0], box[1]], [box[0], box[3]], [box[2], box[3]], [box[2], box[1]]]
        self.poly = Polygon(pts)
        self.radius = max(abs(box[2]-box[0]), abs(box[3]-box[1]))
        self.p = ((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) * 0.25,
            (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) * 0.25)
        self.track_id = None
        self.next_objs = []
        self.next_obj_ious = []
        self.frames = collections.OrderedDict()
        self.matched = False
        self.inactive_count = 0
        self.refined_poly = None
        self.discarded = False

def load_objs(path):
    with open(path) as f:
        objs = []
        for idx, obj_json in enumerate(json.load(f)):
            objs.append(Obj(idx, obj_json))
    return objs

def dist(a, b):
    return math.sqrt((a.p[0]-b.p[0])*(a.p[0]-b.p[0]) + (a.p[1]-b.p[1])*(a.p[1]-b.p[1]))

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

def track(base_path, video_path):
    track_count = 0

    if video_path is not None:
        color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(0,0,0),(255,255,255),
                (180,0,0),(0,180,0),(0,0,180),(180,180,0),(180,0,180),(255,180,255),(255,255,180),
                (180,255,0),(180,0,255),(0,180,255),(255,180,0),(255,0,180),(0,255,180),(180,255,255),
                ]
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Resolution:\t%d x %d" % (frame_w, frame_h))
        print("FPS:\t\t%6.2f" % fps)
        print("Frame Count:\t", int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

        out_video_path = video_path[0:video_path.rfind(".")] + "_tracked.mp4"
        out_video = cv2.VideoWriter(out_video_path,cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), fps, (frame_w,frame_h))
        print(out_video_path)

    active_tracked_objs = collections.OrderedDict()
    inactive_tracked_objs = collections.OrderedDict()
    img = None
    frame_id = -1
    files = sorted(os.listdir(base_path))
    json_frame_id_old = -1
    start = time.time()
    for item in files:
        if item.find(".json") == -1:
            continue

        tokens = item.replace(".json", "").split("_")
        json_frame_id = int(tokens[0])
        angle = int(tokens[1]) if len(tokens) > 1 else 0
        
        if angle not in {0, 10, 20, 30, 40, 50, 60, 70, 80}:
        #if angle not in {0, 15, 45, 60, 75}:
            continue

        if 0:
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
        
        is_new_frame = json_frame_id_old != -1 and json_frame_id_old != json_frame_id
        if is_new_frame:
            print("[%05d]: %.3fs" % (json_frame_id, time.time() - start))
            start = time.time()
            
            # Discard overlapped objs with less matched count
            discarded_track_ids = set()
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
                        #print("######## iou3: %f, %d(%d)[%s] vs %d(%d)[%s]" % (iou3, obj_a.track_id, len(obj_a.frames), str(obj_a.p), obj_b.track_id, len(obj_b.frames), str(obj_b.p)))
                        if len(obj_a.frames) > len(obj_b.frames):
                            #print("######## discard b", len(obj_b.frames))
                            discarded_track_ids.add(obj_b.track_id)
                        elif len(obj_a.frames) < len(obj_b.frames):
                            #print("######## discard a", len(obj_b.frames))
                            discarded_track_ids.add(obj_a.track_id)
                        elif obj_a.refined_poly.area < obj_b.refined_poly.area:
                            #print("######## discard b, area ", obj_b.refined_poly.area)
                            discarded_track_ids.add(obj_b.track_id)
                        else:
                            #print("######## discard a, area ", obj_a.refined_poly.area)
                            discarded_track_ids.add(obj_a.track_id)
            #print("######### max_iou3: ", max_iou3)

            for track_id in discarded_track_ids:
                obj = active_tracked_objs[track_id]
                obj.discarded = True
                #print("############################# discard track ", track_id, obj.track_id)
                inactive_tracked_objs[track_id] = obj
                #print(list(active_tracked_objs.keys()))
                del active_tracked_objs[track_id]
                #print(list(active_tracked_objs.keys()))

            # show results
            #if 0:
            if video_path is not None:
                for obj in active_tracked_objs.values():
                    c = color[obj.track_id % len(color)]
                    pts =[]
                    for x,y in obj.refined_poly.exterior.coords:
                        pts.append((x,y))
                    pts = np.int32(np.array(pts))
                    img = cv2.drawContours(img, [pts], 0, c, 3)
                #img = cv2.fillPoly(img, [cd], (255,0,0))
                #img = cv2.rectangle(img, (int(obj.box[0]), int(obj.box[1])), (int(obj.box[2]), int(obj.box[3])), c, 3)
                out_video.write(img)

                cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("result",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.imshow("result", img)
                #img2 = img[1124:1124+103, 1364:1364+176]
                #img2 = cv2.resize(img2, (img2.shape[1]*4, img2.shape[0]*4))
                #cv2.imshow("result", img2)
                #cv2.imwrite("saved.png", img)
                #exit(0)
                k = cv2.waitKey(1)
                if k == 27:
                #    #exit(0)
                    break

        json_frame_id_old = json_frame_id

        if video_path is not None:
            while json_frame_id != frame_id or video_img is None:
                _, video_img = video.read()
                frame_id += 1
            img = video_img.copy()

        json_path = os.path.join(base_path, item)
        objs = load_objs(json_path)

        new_tracks = []
        for obj in active_tracked_objs.values():
            obj.next_objs = []
            obj.next_obj_ious = []

        # Track with IoU
        for obj in objs:
            tracked_obj, iou = max_IoU_obj(obj, active_tracked_objs.values(), IoU_poly_vs_refined_poly)

            if in_range(obj):
                print("iou", iou, json_frame_id, angle)

                if json_frame_id >= 40:
                #if json_frame_id == 17 and angle == 0:
                    c = [255, 255, 255]
                    pts =[]
                    for x,y in obj.poly.exterior.coords:
                        pts.append((x,y))
                    pts = np.int32(np.array(pts))
                    img = cv2.drawContours(img, [pts], 0, c, 1)
                    
                    for obj2 in active_tracked_objs.values():
                        if in_range(obj2):
                            c = [0, 0, 0]
                            pts =[]
                            for x,y in obj2.refined_poly.exterior.coords:
                                pts.append((x,y))
                            pts = np.int32(np.array(pts))
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

            if tracked_obj is None:
                new_tracks.append(obj)
            else:
                obj.track_id = tracked_obj.track_id
                tracked_obj.next_objs.append(obj)
                tracked_obj.next_obj_ious.append(iou)

        # Split track that has multiple next objs in the same frame
        for obj in active_tracked_objs.values():
            if in_range(obj):
                print("len(obj.next_objs) ", len(obj.next_objs))
                
            if len(obj.next_objs) == 0:
                continue
            max_iou = max(obj.next_obj_ious)
            kk = -1
            for k, next_obj in enumerate(obj.next_objs):
                if obj.next_obj_ious[k] == max_iou:
                    selected_obj = next_obj
                    kk = k
            # Ignore highly overlapped detections
            for k, next_obj in enumerate(obj.next_objs):
                if k != kk:
                    iou2 = selected_obj.poly.intersection(next_obj.poly).area / selected_obj.poly.union(next_obj.poly).area
                    if iou2 < 0.9:
                        new_tracks.append(next_obj)
            obj.next_objs = [selected_obj]
            obj.next_obj_ious = [max_iou]

        # Add new tracks for non-matched objs
        for obj in new_tracks:
            if in_range(obj):
                print("Add new track")
            active_tracked_objs[track_count] = obj
            obj.track_id = track_count
            obj.refined_poly = copy.deepcopy(obj.poly)
            track_count += 1
        #print("new_tracks: ", len(new_tracks))
        
        # Store matched results to frames
        to_delete = []
        for obj in active_tracked_objs.values():

            if in_range(obj):
                print("matched ", obj.matched)

            if is_new_frame:
                if not obj.matched:
                    obj.inactive_count += 1
                    if obj.inactive_count >= 8:
                        to_delete.append(obj.track_id)
                    continue
                obj.matched = False

            if len(obj.next_objs) > 0:
                obj.matched = True
                obj.inactive_count = 0
                if json_frame_id not in obj.frames.keys():
                    obj.frames[json_frame_id] = collections.OrderedDict()
                    obj.refined_poly = copy.deepcopy(obj.next_objs[0].poly)
                    obj.p = obj.next_objs[0].p
                else:
                    obj.refined_poly = obj.refined_poly.intersection(obj.next_objs[0].poly)
                obj.frames[json_frame_id][angle] = obj.next_objs[0]

        #print("to delete: ", len(to_delete))
        # Remove objs that has been inactive for a while
        for track_id in to_delete:
            inactive_tracked_objs[track_id] = active_tracked_objs[track_id]
            del active_tracked_objs[track_id]

        if 0:
            # save to json
            frame_json = []
            for obj in objs2:
                obj.json["obj_id"] = obj.track_id
                frame_json.append(obj.json)
            with open(json_path, 'w') as f:
                json.dump(frame_json, f, indent=4)

        if 0:
            max_dist = 0
            max_dist_iou = None
            for obj in objs1:
                if len(obj.next_objs) > 0:
                    d = dist(obj, obj.next_objs[0])
                    if d > max_dist:
                        max_dist = d
                        max_dist_iou = obj.next_obj_ious[0]
            print("[%05d]: dist: %10.2f, %.3f" % (len(frames), max_dist, max_dist_iou))

        if 0:
            fig, ax = plt.subplots(figsize=(15,7))
            for obj in objs1:
                ax.plot(obj.p[0], obj.p[1], 'bo', markersize = 3)
            for obj in objs2:
                ax.plot(obj.p[0], obj.p[1], 'rs',  markersize = 2)

            for obj in objs1:
                if len(obj.next_objs) > 0:
                    a = [obj.p[0], obj.next_objs[0].p[0]]
                    b = [obj.p[1], obj.next_objs[0].p[1]]
                    ax.plot(a, b, 'k')

            plt.show()
            if 0:
                fig.canvas.draw()
                img = np.array(fig.canvas.renderer._renderer)
                #cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
                #cv2.setWindowProperty("result",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.imshow("result", img)
                k = cv2.waitKey()
                if k == 27:
                    exit(0)

    if video_path is not None:
        out_video.release()
        video.release()
        print(len(active_tracked_objs) + len(inactive_tracked_objs))

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python track.py detection-folder [video.mp4]")
    else:
        track(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)