import os
import math
import sys
import json
import cv2
import collections
import bisect
import pprint

def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def dot(p1, p2):
    return p1[0]*p2[0] + p1[1]*p2[1]

def deg2rad(v):
    return v / 180.0 * 3.14159265

def rad2deg(v):
    return v / 3.14159265 * 180.0

class State:
    def __init__(self, frame_id, time, obj_json):
        self.frame_id = frame_id
        self.time = time

        #polyon = obj_json["fixed_polygon"]
        polyon = obj_json["remapped_polygon_in_local_ENU"]
        self.x = (polyon[0][0] + polyon[1][0] + polyon[2][0] + polyon[3][0]) * 0.25
        self.y = (polyon[0][1] + polyon[1][1] + polyon[2][1] + polyon[3][1]) * 0.25

        # Use angle of min area bounding box as the heading_angle of vehicles
        angle = obj_json["angle"]
        #length = obj_json["length"]
        #width = obj_json["width"]
        if angle != 9999.0:
            self.width = math.sqrt(dist2(polyon[0], polyon[1]))
            self.length = math.sqrt(dist2(polyon[1], polyon[2]))
            self.heading_angle = rad2deg(math.atan2(polyon[2][1]-polyon[1][1], polyon[2][0]-polyon[1][0]))
            if self.length < self.width:
                self.length, self.width = self.width , self.length  # swap
                self.heading_angle += 90.0
            while self.heading_angle > 360.0:
                self.heading_angle -= 360.0
            while self.heading_angle < 0.0:
                self.heading_angle += 360.0
        else:
            self.heading_angle = None
            self.length = None
            self.width = None
        self.aabb = obj_json["aabb"]

class Obj:
    def __init__(self, id):
        self.id = id
        self.type = 1
        self.states = []

def generate(objs_base_path, video_path, csv_path, out_json_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()

    objs = collections.OrderedDict()
    frames = collections.OrderedDict()
    print(objs_base_path)
    files = list(sorted(os.listdir(objs_base_path)))
    for item in files:
        if item.find(".tracked.json") == -1:
            continue
        json_frame_id = int(item.replace(".tracked.json", ""))
        json_path = os.path.join(objs_base_path, item)
        print(json_path)
        with open(json_path) as f:
            frame = json.loads(f.read().replace("'", "\""))

        frames[json_frame_id] = []

        time = json_frame_id / fps
        for obj_json in frame:
            id = obj_json["obj_id"]
            #if obj_json["angle"] == 9999.0:
            #    continue
            if id not in objs:
                objs[id] = Obj(id)
            objs[id].states.append(State(json_frame_id, time, obj_json))

    markers = collections.OrderedDict()

    for obj_id, obj in objs.items():
        for s in obj.states:
            if s.heading_angle is None or s.length is None or s.width is None:
                continue
            if obj_id not in markers:
                markers[obj_id] = collections.OrderedDict()
            markers[obj_id][s.frame_id] = {"id":obj_id, "frame_id":s.frame_id, "x":round(s.x,4), "y":round(s.y,4),
                "length":round(s.length,4), "width":round(s.width,4), "heading_angle":round(s.heading_angle,4), "manually_keyed": True}
        
        if obj_id not in markers:
            # The trajectory is too short. No angle values are available.
            s = obj.states[0]
            markers[obj_id] = collections.OrderedDict({s.frame_id: {"id":obj_id, "frame_id":s.frame_id, "x":round(s.x,4), "y":round(s.y,4),
                "length":1.0, "width":1.0, "heading_angle":45.0, "manually_keyed": True}})
        else:
            # Correct heading directions assuming cars are moving forward.
            keys = list(markers[obj_id].keys())
            for k, marker in enumerate(markers[obj_id].values()):
                rad = deg2rad(marker["heading_angle"])
                v = (math.cos(rad), math.sin(rad))
                vv = [0.0, 0.0]
                d2 = 0.0
                n = 0
                p1 = [0.0, 0.0]
                p2 = [0.0, 0.0]
                while d2 < 100.0 and (k-n >= 0 or k+n < len(markers[obj_id])):
                    k0 = keys[max(0, k-n)]
                    k1 = keys[min(len(markers[obj_id])-1, k+n)]
                    p1[0] = markers[obj_id][k0]["x"]
                    p1[1] = markers[obj_id][k0]["y"]
                    p2[0] = markers[obj_id][k1]["x"]
                    p2[1] = markers[obj_id][k1]["y"]
                    d2 = dist2(p1, p2)
                    vv[0] = p2[0] - p1[0]
                    vv[1] = p2[1] - p1[1]
                    n += 1
                if dot(v, vv) < 0.0:
                    heading_angle = marker["heading_angle"] + 180.0
                    if heading_angle > 360.0:
                        heading_angle -= 360.0
                    marker["heading_angle"] = heading_angle

    if out_json_path is not None:
        with open(out_json_path, "w") as f:
            f.write(pprint.pformat([list(marker.values())for marker in markers.values()], width=200, indent=1).replace("'", "\"").replace("True", "true"))

    # Export as standard raw data in CSV
    with open(csv_path, "w") as f:
        #title = "ObjID, Type, Time[s], FrameID, PositionX[m], PositionY[m], Length[m], Width[m], HeadingAngle[degree], MinX[pixel], MinY[pixel], MaxX[pixel], MaxY[pixel]"
        title = "ID,Time,PositionX,PositionY,Length,Width,Height,Yaw,Category,Style,Color,VX,VY,AX,AY"
        f.write(title + "\n")
        for obj_id, obj in objs.items():
            frame_ids = list(markers[obj_id].keys())
            for s in obj.states:
                obj_type = 0
                frame_idx2 = bisect.bisect_right(frame_ids, s.frame_id)
                frame_idx1 = frame_idx2 -1
                if frame_idx2 == 0:
                    heading_angle = markers[obj_id][frame_ids[0]]["heading_angle"]
                    length = markers[obj_id][frame_ids[0]]["length"]
                    width = markers[obj_id][frame_ids[0]]["width"]
                elif frame_idx2 == len(frame_ids):
                    heading_angle = markers[obj_id][frame_ids[-1]]["heading_angle"]
                    length = markers[obj_id][frame_ids[-1]]["length"]
                    width = markers[obj_id][frame_ids[-1]]["width"]
                else:   # linear interpolated length, width and heading_angle values
                    frame_id1 = frame_ids[frame_idx1]
                    frame_id2 = frame_ids[frame_idx2]
                    a = float(s.frame_id - frame_id1) / (frame_id2 - frame_id1)
                    m1 = markers[obj_id][frame_id1]
                    m2 = markers[obj_id][frame_id2]
                    length = m1["length"] * (1-a) + m2["length"] * a
                    width = m1["width"] * (1-a) + m2["width"] * a
                    delta_angle = m2["heading_angle"] - m1["heading_angle"]
                    delta_angle = delta_angle % 360.0
                    delta_angle = (delta_angle + 540.0) % 360.0 - 180.0
                    heading_angle = m1["heading_angle"] + delta_angle * a
                # Use index as ID
                #f.write("%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n" % (obj_id, obj_type, s.frame_id, s.time, round(s.x,4), round(s.y,4), round(length,4), round(width,4), round(heading_angle,4), s.aabb[0][0], s.aabb[0][1], s.aabb[1][0], s.aabb[1][1]))
                f.write("%d,%.4f,%.4f,%.4f,%.4f,%.4f,0.0,%.4f,,,,,,,\n" % (obj_id, s.time, round(s.x,4), round(s.y,4), round(length,4), round(width,4), round(heading_angle,4)))

if __name__ == '__main__':
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print("Usage: python generate_standard_raw_data.py detection-folder video.mp4 out.csv [out.vehicle_markers.json]")
    else:
        generate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) == 5 else None)
