import time
import os
import sys
import math
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def test():
    np.random.seed(100)

    points1 = np.array([(x, y) for x in np.linspace(-1,1,7) for y in np.linspace(-1,1,7)])
    N = points1.shape[0]
    points2 = 2*np.random.rand(N,2)-1

    C = cdist(points1, points2)

    _, assigment = linear_sum_assignment(C)

    plt.plot(points1[:,0], points1[:,1],'bo', markersize = 10)
    plt.plot(points2[:,0], points2[:,1],'rs',  markersize = 7)
    for p in range(N):
        plt.plot([points1[p,0], points2[assigment[p],0]], [points1[p,1], points2[assigment[p],1]], 'k')
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)
    plt.axes().set_aspect('equal')
    plt.show()

def IoU(b1, b2):
    intersection = [max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])]
    if intersection[2] < intersection[0] or intersection[3] < intersection[1]:
        return 0.0
    intersection_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
    union_area = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - intersection_area
    #print(intersection_area, union_area)
    return intersection_area / union_area

class Obj:
    def __init__(self, box):
        self.id = id
        self.box = box
        self.p = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
        self.track_id = None
        self.next_objs = []
        self.next_obj_ious = []
    #def __str__(self):
    #    return "[%s]: %d, %d, %d, %d; %s" % (self.id, self.left, self.top, self.right, self.bottom, str(self.obb))

class Frame:
    def __init__(self, path):
        with open(path) as f:
            self.objs = []
            for obj in json.load(f):
                self.objs.append(Obj(obj["box"]))

def dist(a, b):
    return math.sqrt((a.p[0]-b.p[0])*(a.p[0]-b.p[0]) + (a.p[1]-b.p[1])*(a.p[1]-b.p[1]))

def max_IoU_obj(obj_a, objs):
    max_IoU = 0.2
    max_IoU_obj = None
    for obj_b in objs:
        iou = IoU(obj_a.box, obj_b.box)
        if iou > max_IoU:
            max_IoU = iou
            max_IoU_obj = obj_b
            #print(iou)
    return max_IoU_obj, max_IoU

def track(base_path, video_path):
    frames = []
    track_count = 0
    color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(0,0,0),(255,255,255),
            (127,0,0),(0,127,0),(0,0,127),(127,127,0),(127,0,127),(0,127,127),(255,127,255),(255,255,127),
            (127,255,0),(127,0,255),(0,127,255),(255,127,0),(255,0,127),(0,255,127),(127,255,255),
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

    frame_id = -1
    for item in os.listdir(base_path):
        if item.find(".json") != -1:
            start = time.time()

            json_frame_id = int(item.replace(".json", ""))
            while json_frame_id != frame_id:
                _, img = video.read()
                frame_id += 1

            frames.append(Frame(os.path.join(base_path, item)))

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
                if len(obj.next_objs) > 1:
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
            if 1:
 
                for obj in objs2:
                    c = color[obj.track_id % len(color)]
                    img = cv2.rectangle(img, (int(obj.box[0]), int(obj.box[1])), (int(obj.box[2]), int(obj.box[3])), c, 4)

                cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("result",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.imshow("result", img)
                k = cv2.waitKey(1)
                if k == 27:
                    #exit(0)
                    break
                out_video.write(img)

            print("[%05d]: %.3fs" % (len(frames), time.time() - start))

    out_video.release()
    video.release()
    print(len(frames))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python track.py detection-folder video.mp4")
    else:
        track(sys.argv[1], sys.argv[2])