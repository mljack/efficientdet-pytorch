# coding=utf-8

import json

def str2mat(s):  
    ss = []
    for c in s.strip('[').strip(']').split(';'):
        ss.append([float(x) for x in c.split(',')])
    return ss

def add_common_args(parser):
    parser.add_argument("--input-file", type=str, dest="input", default="test/64_20200529172236_offset.csv", help="(str,test/64_20200529172236_offset.csv,%%s,#Input Trajectories)")
    parser.add_argument("--output-file", type=str, dest="output", default="", help="(str,,%%s,#Output Trajectories)")
    parser.add_argument("--traj-type", type=str, dest="traj_type", default="unified", help="(str,,%%s,#Trajectory Type(unified|main|objs)")

def read_json(path):
    with open(path) as f:
        return json.load(f)
    return None

def write_json(path, j):
    with open(path, "w") as f:
        json.dump(j, f, indent=4)

