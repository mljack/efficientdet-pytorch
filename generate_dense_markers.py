import os
import math
import sys
import json
import cv2
import collections
import bisect
import pprint

def dist2(p1, p2):
    return (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1])

def dot(p1, p2):
    return p1[0]*p2[0] + p1[1]*p2[1]


class State:
    def __init__(self, frame_id, time, state):
        self.frame_id = frame_id
        self.time = time
        center = state["center"]
        self.x = center[0]
        self.y = center[1]
        if state["angle"] != 9999.0:
            polyon = state["poly"]
            self.width = math.sqrt(dist2(polyon[0], polyon[1]))
            self.length = math.sqrt(dist2(polyon[1], polyon[2]))
            self.heading_angle = math.degrees(math.atan2(polyon[2][1]-polyon[1][1], polyon[2][0]-polyon[1][0]))
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

class Obj:
    def __init__(self, id):
        self.id = id
        self.type = 1
        self.states = []

def generate(objs_base_path, video_path, csv_path):
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
            if id not in objs:
                objs[id] = Obj(id)
            objs[id].states.append(State(json_frame_id, time, obj_json))

    markers = collections.OrderedDict()

    for idx, obj in enumerate(objs.values()):
        for s in obj.states:
            if s.heading_angle is None or s.length is None or s.width is None:
                continue
            if s.heading_angle == 9999.0 or s.length == 9999.0 or s.width == 9999.0:
                continue
            if idx not in markers:
                markers[idx] = collections.OrderedDict()
            markers[idx][s.frame_id] = {"id":idx, "frame_id":s.frame_id, "x":round(s.x,4), "y":round(s.y,4),
                "length":round(s.length,4), "width":round(s.width,4), "heading_angle":round(s.heading_angle,4), "manually_keyed": True}
        
        # The trajectory is too short. No angle values are available.
        if idx not in markers:
            s = obj.states[0]
            markers[idx] = collections.OrderedDict({s.frame_id: {"id":idx, "frame_id":s.frame_id, "x":round(s.x,4), "y":round(s.y,4),
                "length":50.0, "width":24.0, "heading_angle":45.0, "manually_keyed": True}})
        else:
            keys = list(markers[idx].keys())
            for k, marker in enumerate(markers[idx].values()):
                rad = math.radians(marker["heading_angle"])
                v = (math.cos(rad), math.sin(rad))
                vv = [0.0, 0.0]
                d2 = 0.0
                n = 0
                p1 = [0.0, 0.0]
                p2 = [0.0, 0.0]
                while d2 < 100.0 and (k-n >= 0 or k+n < len(markers[idx])):
                    k0 = keys[max(0, k-n)]
                    k1 = keys[min(len(markers[idx])-1, k+n)]
                    p1[0] = markers[idx][k0]["x"]
                    p1[1] = markers[idx][k0]["y"]
                    p2[0] = markers[idx][k1]["x"]
                    p2[1] = markers[idx][k1]["y"]
                    d2 = dist2(p1, p2)
                    vv[0] = p2[0] - p1[0]
                    vv[1] = p2[1] - p1[1]
                    n += 1
                if dot(v, vv) < 0.0:
                    heading_angle = marker["heading_angle"] + 180.0
                    if heading_angle > 360.0:
                        heading_angle -= 360.0
                    marker["heading_angle"] = heading_angle

    #with open(csv_path.replace(".csv", ".vehicle_markers.json"), "w") as f:
    #    f.write(pprint.pformat([list(marker.values())for marker in markers.values()], width=200, indent=1).replace("'", "\"").replace("True", "true"))

    # Export to data-from-sky CSV for CyTrafficEditor
    with open(csv_path, "w") as f:
        title = "Track ID, Type, Entry Gate, Entry Time [s], Exit Gate, Exit Time [s], Traveled Dist. [px], Avg. Speed [kpx/h], Trajectory(x [px], y [px], Speed [kpx/h], Total Acc. [pxs-2], Time [s], Heading Angle, Length, Width, )"
        #title = "Track ID, Type, Entry Gate, Entry Time [s], Exit Gate, Exit Time [s], Traveled Dist. [px], Avg. Speed [kpx/h], Trajectory(x [px], y [px], Speed [kpx/h], Total Acc. [pxs-2], Time [s], )"
        obj_attr = ", Car, - , - , - , - , - , - , "
        f.write(title + "\n")
        for idx, obj in enumerate(objs.values()):
            f.write(str(idx) + obj_attr)    # Use index as ID
            frame_ids = list(markers[idx].keys())
            for s in obj.states:
                frame_idx2 = bisect.bisect_right(frame_ids, s.frame_id)
                frame_idx1 = frame_idx2 -1 
                if frame_idx2 == 0:
                    heading_angle = markers[idx][frame_ids[0]]["heading_angle"]
                    length = markers[idx][frame_ids[0]]["length"]
                    width = markers[idx][frame_ids[0]]["width"]
                elif frame_idx2 == len(frame_ids):
                    heading_angle = markers[idx][frame_ids[-1]]["heading_angle"]
                    length = markers[idx][frame_ids[-1]]["length"]
                    width = markers[idx][frame_ids[-1]]["width"]
                else:   # linear interpolate length, width and heading_angle values
                    frame_id1 = frame_ids[frame_idx1]
                    frame_id2 = frame_ids[frame_idx2]
                    a = float(s.frame_id - frame_id1) / (frame_id2 - frame_id1)
                    m1 = markers[idx][frame_id1]
                    m2 = markers[idx][frame_id2]
                    length = m1["length"] * (1-a) + m2["length"] * a
                    width = m1["width"] * (1-a) + m2["width"] * a
                    delta_angle = m2["heading_angle"] - m1["heading_angle"]
                    delta_angle = delta_angle % 360.0
                    delta_angle = (delta_angle + 540.0) % 360.0 - 180.0
                    heading_angle = m1["heading_angle"] + delta_angle * a
                f.write("%.4f, %.4f, 0.0, 0.0, %.4f, %.4f, %.4f ,%.4f, " % (round(s.x,4), round(s.y,4), s.time, round(heading_angle,4), round(length,4), round(width,4)))
            f.write("\n")

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python generate_dense_markers.py detection-folder video.mp4 out.csv")
    else:
        generate(sys.argv[1], sys.argv[2], sys.argv[3])
