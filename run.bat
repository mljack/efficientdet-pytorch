date /t >> zdj_log.txt
time /t >> zdj_log.txt

python infer.py _datasets\test\zdj.mp4
python track.py _datasets\test\zdj_objs _datasets\test\zdj.mp4
python generate_traffichut_csv.py _datasets\test\zdj_objs _datasets\test\zdj.mp4 zdj.csv

date /t >> zdj_log.txt
time /t >> zdj_log.txt
