import os
import math
import sys
import json
import cv2
import collections

'''
id,type,time,x_bbox,y_bbox,w_bbox,h_bbox,xcenter,ycenter,speed,arctan
0,1,0.08,2756,0,31,14,2771,7,25.0,0
0,1,0.12,2756,0,29,12,2770,6,0.0,0
0,1,0.16,2755,0,30,12,2770,6,25.0,0
'''

class State:
    def __init__(self, time, box):
        self.time = time
        self.box_x = min(box[0][0], box[2][0])
        self.box_y = min(box[0][1], box[2][1])
        self.box_w = abs(box[2][0] - box[0][0])
        self.box_h = abs(box[2][1] - box[0][1])
        self.x = self.box_x+self.box_w/2
        self.y = self.box_y+self.box_h/2
        self.speed = 0
        self.arctan = 0

class Obj:
    def __init__(self, id):
        self.id = id
        self.type = 1
        self.states = []

def generate(objs_base_path, video_path, csv_path):
    mask_path = video_path.replace(".mp4", ".png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    objs = collections.OrderedDict()
    frames = collections.OrderedDict()

    files = list(sorted(os.listdir(objs_base_path)))
    for item in files:
        if item.find(".tracked.json") == -1:
            continue
        json_frame_id = int(item.replace(".tracked.json", "").split("_")[0]) + 1
        json_path = os.path.join(objs_base_path, item)
        print(json_path)
        with open(json_path) as f:
            frame = json.loads(f.read().replace("'", "\""))

        frames[json_frame_id] = []

        time = json_frame_id / fps
        for obj_json in frame:
            id = obj_json["obj_id"]
            if id not in objs:
                objs[id] = Obj(id)
            objs[id].states.append(State(time, obj_json["boxes"][0]["polygon"]))

    n = 8
    for obj in objs.values():
        for i, state in enumerate(obj.states):
            if state.box_w > 120 or state.box_h > 120:
                obj.type = 2
            i0 = max(0, i - n)
            i1 = min(len(obj.states)-1, i + n)
            if i0 < i1:
                dx = obj.states[i1].x - obj.states[i0].x
                dy = obj.states[i1].y - obj.states[i0].y
                dt = obj.states[i1].time - obj.states[i0].time
                state.speed = math.sqrt(dx*dx+dy*dy)/dt
                state.arctan = math.atan2(dx, -dy)
            state.id = obj.id
            frames[int(round(state.time * fps))].append(state)

    if 1:
        with open(csv_path, "w") as f:
            f.write("id,type,time,x_bbox,y_bbox,w_bbox,h_bbox,xcenter,ycenter,speed,arctan\n")
            for obj in objs.values():
                lines = []
                for s in obj.states:
                    if mask is not None and (s.x < 0.0 or s.y < 0.0 or s.x >= mask.shape[1] or s.y >= mask.shape[0] or mask[int(s.y)][int(s.x)] == 0):
                        continue
                    if abs((s.time + 1e-6) % 0.2) > 1e-3:
                        continue
                    lines.append([s.time, int(s.box_x), int(mask.shape[0]-1-s.box_y), int(s.box_w), int(s.box_h), int(s.x), int(mask.shape[0]-1-s.y), round(s.speed, 2), round(s.arctan, 2)])
                if len(lines) > 0:
                    f.write(str(obj.id)+","+str(obj.type)+",")
                    for line in lines:
                        f.write(",".join([str(t) for t in line]))
                        f.write(";")
                    f.write("\n")

    if 1:
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if sys.platform == "win32":
            fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(video_path.replace(".", "_tracked."), fourcc, fps, (frame_w,frame_h))
        color = ((255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(0,0,0),(255,255,255),
                (180,0,0),(0,180,0),(0,0,180),(180,180,0),(180,0,180),(255,180,255),(255,255,180),
                (180,255,0),(180,0,255),(0,180,255),(255,180,0),(255,0,180),(0,255,180),(180,255,255))
        count = 0
        has_frame = True
        while has_frame:
            count += 1
            has_frame, img = video.read()
            if not has_frame:
                break
            if count not in frames and count > 10:
                print("No more data after frame %d!" % count)
                break
            if count in frames:
                for s in frames[count]:
                    if mask is not None and (s.x < 0.0 or s.y < 0.0 or s.x >= mask.shape[1] or s.y >= mask.shape[0] or mask[int(s.y)][int(s.x)] == 0):
                        continue
                    img = cv2.rectangle(img, (int(s.box_x), int(s.box_y)), (int(s.box_x+s.box_w), int(s.box_y+s.box_h)), color[s.id % len(color)], 3)
                    img = cv2.line(img, (int(s.x), int(s.y)), (int(s.x-s.speed*math.sin(-s.arctan)), int(s.y-s.speed*math.cos(-s.arctan))), (0,255,0), 3)

            out_video.write(img)
            if 1:
                cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("result",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.imshow("result", img)
                #cv2.imwrite("saved.png", img)
                k = cv2.waitKey(1)
                if k == 27:
                    break
        out_video.release()
    video.release()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python generate_traffichut.py detection-folder video.mp4 out.csv")
    else:
        generate(sys.argv[1], sys.argv[2], sys.argv[3])
