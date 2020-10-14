import sys
import os
import random

base = sys.argv[1]
all_jpg = []
for item in os.listdir(base):
    if item.find(".jpg") == -1:
        continue
    all_jpg.append(os.path.join(base, item))

random.shuffle(all_jpg)

total = len(all_jpg)
train_n = int(total * 0.8)
train_set = []
test_set = []
if os.path.isfile("train.txt"):
    with open("train.txt") as f:
        train_set = f.readlines()
if os.path.isfile("test.txt"):
    with open("test.txt") as f:
        test_set = f.readlines()

for i, item in enumerate(all_jpg):
    if i < train_n:
        train_set.append(item + "\n")
    else:
        test_set.append(item + "\n")

random.shuffle(train_set)
random.shuffle(test_set)

with open("train.txt", "w") as f:
    for line in train_set:
        f.write(line)

with open("test.txt", "w") as f:
    for line in test_set:
        f.write(line)
