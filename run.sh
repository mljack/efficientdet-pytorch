video="syq1"
echo ${video}
date "+%F %H:%M:%S" >> ${video}_log.txt
python infer.py _videos/${video}.mp4
python track.py _videos/${video}_objs _videos/${video}.mp4
python generate_traffichut_csv.py _videos/${video}_objs _videos/${video}.mp4 ${video}.csv
date "+%F %H:%M:%S" >> ${video}_log.txt

video="jyq4"
echo ${video}
date "+%F %H:%M:%S" >> ${video}_log.txt
python infer.py _videos/${video}.mp4
python track.py _videos/${video}_objs _videos/${video}.mp4
python generate_traffichut_csv.py _videos/${video}_objs _videos/${video}.mp4 ${video}.csv
date "+%F %H:%M:%S" >> ${video}_log.txt

#python track.py _videos/sf6_objs _videos/sf6.mp4
#python track.py _videos/jy3_objs _videos/jy3.mp4
#python track.py _videos/sy4_objs _videos/sy4.mp4
#python track.py _videos/ys1_objs _videos/ys1.mp4
#python infer.py _videos/gm7.mp4
#python track.py _videos/gm7_objs _videos/gm7.mp4

