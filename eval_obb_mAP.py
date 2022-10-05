import os
import shutil

common_vehicle_width = 25.0
models = [
    #"_models/model-005-best-checkpoint-000epoch.bin",
    #"_models/model-007-best-checkpoint-003epoch.bin",
    #"_models/model-013-best-checkpoint-001epoch.bin",
    #"_models/model-018-best-checkpoint-001epoch.bin",
    #"_models/model-021-best-checkpoint-002epoch.bin",
    #"_models/model-023-best-checkpoint-000epoch.bin",
    "_models/model-073-best-checkpoint-036epoch.bin",
    "_models/model-092-best-checkpoint-035epoch.bin",
    "_models/model-z0009-best-checkpoint-054epoch.bin",
    #"_models/model-z0011-best-checkpoint-062epoch.bin",
]

test_set = "efficientdet_pytorch_win64/_datasets/_test_sets/private170"
#test_set = "efficientdet_pytorch_win64/_datasets/_test_sets/TrafficHUT2020"
#test_set = "efficientdet_pytorch_win64/_datasets/_test_sets/TrafficHUT2020_768"

for model_path in models:
    gt_path = test_set
    det_path = f"{test_set}_det"
    short_id = model_path.replace("_models/", "").replace("-best-checkpoint", "").replace(".bin", "")
    det_path2 = f"{test_set}_det({short_id})"
    if not os.path.exists(det_path2):
        if os.path.exists(det_path):
            shutil.rmtree(det_path)
        os.mkdir(det_path)

        #infer_cmd = "python -m efficientdet_pytorch_win64.infer %s %.1f %s" % (test_set, -1, model_path)
        infer_cmd = "python -m efficientdet_pytorch_win64.infer %s %.1f %s obb" % (test_set, common_vehicle_width, model_path)
        #infer_cmd = "python -m efficientdet_pytorch_win64.infer4 %s %.1f %s" % (test_set, common_vehicle_width, model_path)
        os.system(infer_cmd)
        os.rename(det_path, det_path2)
    else:
        print(f"Found det results: [{det_path2}]")

for model_path in models:
    gt_path = test_set
    short_id = model_path.replace("_models/", "").replace("-best-checkpoint", "").replace(".bin", "")
    det_path2 = f"{test_set}_det({short_id})"
    eval_cmd = f"python Object-Detection-Metrics/pascalvoc.py -gt ../{gt_path} -det ../{det_path2} -detformat obb_json -gtformat obb_json"
    print(short_id)
    os.system(eval_cmd)
