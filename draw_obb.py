import sys
import os
import json
import math
import cv2
import numpy as np

def run(image_folder, marker_folder):
    for item in os.listdir(image_folder):
        name, ext = os.path.splitext(item)
        img_path = os.path.join(image_folder, item)
        if not os.path.exists(img_path) or ext.lower() not in {".jpg", ".jpeg", ".bmp", ".png"}:
            continue
        marker_path = os.path.join(marker_folder, f"{name}.vehicle_markers.json")
        if not os.path.exists(marker_path):
            continue
        print(marker_path)
        with open(marker_path) as f:
            j = json.load(f)
        img = cv2.imread(img_path)
        for obj in j:
            obj = obj[0]
            p = np.array((obj["x"], obj["y"]))
            width = obj["width"]
            length = obj["length"]
            yaw = math.radians(obj["heading_angle"])
            dir = np.array((math.cos(yaw), math.sin(yaw))) * 0.5 * length
            normal = np.array((-math.sin(yaw), math.cos(yaw))) * 0.5 * width
            pts = [
                p - dir - normal,
                p + dir - normal,
                p + dir + normal,
                p - dir + normal,
            ]
            if "enabled" in obj and not obj["enabled"]:
                continue
            if 1:
                certainty = obj["certainty"] if "certainty" in obj else 1.0
                confidence = obj["score"]
            else: # for gt
                certainty = 1.0
                confidence = 1.0
            c = (0, 0, 255)
            if confidence < 0.5:
                c = (0, 255, 255)
            elif confidence < 0.7:
                c = (0, 255, 0)
            if 0.75 > certainty > 0.3:
                c = (255, 255, 0)

            #print(obj)
            img = cv2.drawContours(img, np.int32([pts]), 0, c, 2)
        out_img_path = os.path.join(marker_folder, f"{name}.jpg")
        cv2.imwrite(out_img_path, img)
        if 0:
            #cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
            #cv2.setWindowProperty("test",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("test", img)
            k = cv2.waitKey(0)
            if k == 27:
                exit(0)
        #exit(0)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python draw_obb.py image_folder det_marker_folder")
    else:
        run(sys.argv[1], sys.argv[2])