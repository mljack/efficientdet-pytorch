import time
import os
import sys
import math
import json
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

def IoU_poly(obj1, obj2):
    dx = abs(obj1.p[0] - obj2.p[0])
    dy = abs(obj1.p[1] - obj2.p[1])
    radius = obj1.radius + obj2.radius
    if dx > radius or dy > radius:
        return 0.0
    intersection_area = obj1.poly.intersection(obj2.poly).area
    if intersection_area == 0.0:
        return 0.0
    return intersection_area / obj1.poly.union(obj2.poly).area

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

class Frame:
    def __init__(self, path):
        with open(path) as f:
            self.objs = []
            for idx, obj_json in enumerate(json.load(f)):
                self.objs.append(Obj(idx, obj_json))

def dist(a, b):
    return math.sqrt((a.p[0]-b.p[0])*(a.p[0]-b.p[0]) + (a.p[1]-b.p[1])*(a.p[1]-b.p[1]))

def max_IoU_obj(obj_a, objs):
    max_IoU = 0.2
    max_IoU_obj = None
    for obj_b in objs:
        iou = IoU_poly(obj_a, obj_b)
        if iou > max_IoU:
            max_IoU = iou
            max_IoU_obj = obj_b
    #print(max_IoU)
    return max_IoU_obj, max_IoU

def track(base_path, video_path):
    frames = []
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

    img = None
    frame_id = -1
    files = sorted(os.listdir(base_path))
    for item in files:
        if item.find(".json") == -1:
            continue

        tokens = item.replace(".json", "").split("_")
        json_frame_id = int(tokens[0])
        angle = int(tokens[1]) if len(tokens) > 1 else 0
        #if angle != 0:
        #    continue

        #if img is not None:
        if img is not None and angle == 0:
            cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("result",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("result", img)
            #cv2.imwrite("saved.png", img)
            #exit(0)
            k = cv2.waitKey(1)
            if k == 27:
                #exit(0)
                break

        start = time.time()
        if video_path is not None:
            while json_frame_id != frame_id or video_img is None:
                _, video_img = video.read()
                frame_id += 1
            img = video_img#.copy()

        json_path = os.path.join(base_path, item)
        frames.append(Frame(json_path))

        objs1 = [] if len(frames) < 2 else frames[-2].objs
        objs2 = frames[-1].objs

        for obj in objs2:
            tracked_obj, iou = max_IoU_obj(obj, objs1)
            if tracked_obj is None:
                obj.track_id = track_count
                track_count += 1
            else:
                if tracked_obj.track_id is None:
                    tracked_obj.track_id = track_count
                    track_count += 1
                obj.track_id = tracked_obj.track_id
                tracked_obj.next_objs.append(obj)
                tracked_obj.next_obj_ious.append(iou)

        # split track with multiple objs in the same frame
        for obj in objs1:
            if len(obj.next_objs) == 0:
                continue
            max_iou = max(obj.next_obj_ious)
            for k, next_obj in enumerate(obj.next_objs):
                if obj.next_obj_ious[k] == max_iou:
                    selected_obj = next_obj
                else:
                    next_obj.track_id = track_count
                    track_count += 1
            obj.next_objs = [selected_obj]
            obj.next_obj_ious = [max_iou]

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
            for obj in objs2:
                c = color[obj.track_id % len(color)]

                pts =[]
                for x,y in obj.poly.exterior.coords:
                    pts.append((x,y))
                pts = np.int32(np.array(pts))
                img = cv2.drawContours(img, [pts], 0, c, 3)
                #img = cv2.fillPoly(img, [cd], (255,0,0))

                #img = cv2.rectangle(img, (int(obj.box[0]), int(obj.box[1])), (int(obj.box[2]), int(obj.box[3])), c, 3)
            out_video.write(img)

        print("[%05d]: %.3fs" % (json_frame_id, time.time() - start))

    if video_path is not None:
        out_video.release()
        video.release()
        print(len(frames))

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python track.py detection-folder [video.mp4]")
    else:
        track(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)