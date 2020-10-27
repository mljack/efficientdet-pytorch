import sys
import cv2
import standard_raw_data


def generate(objs_base_path, video_path, csv_path):

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()

    raw_data_holder = StandardRawDataHolder()

    files = sorted(os.listdir(objs_base_path))
    for item in files:
        print(item)
        if item.find(".json") == -1:
            continue
        json_frame_id = int(item.replace(".json", ""))
        json_path = os.path.join(objs_base_path, item)
        with open(path) as f:
            frame = json.load(f)

        time = json_frame_id * fps
        for obj in frame.objs:
            if obj.track_id not in raw_data_holder.objs.keys():
                print(obj.track_id)
                obj_traj = Trajectory()
                str_id = str(obj.track_id)
                obj_traj.id = str_id
                raw_data_holder.objs[str_id] = obj_traj
                if time in objs_holder.objs[str_id].seq:
                    print("Found data with exactly the same timestamp: id=%s, time=%f" % (str_id, time))
            state = RigidBodyState()
            state.time = time
            state.p = []
            state.v = [0.0,0.0,0.0]
            state.category = "vehicle"
            state.length = 5.0
            state.width = 2.5
            state.height = 1.9
            state.yaw = 0.0
            state.ego = 1 if obj.track_id == 0 else 0
            obj_traj.seq[time] = state
    raw_data_holder.write(csv_path)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python track.py detection-folder video.mp4 output.csv")
    else:
        generate(sys.argv[1], sys.argv[2], sys.argv[3])
