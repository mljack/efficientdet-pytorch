# coding=utf-8
import os
import collections
import csv
import numpy as np
import bisect
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from pyquaternion import Quaternion
from op_common import *
import math

FIELD_DEF = [
    "ID,s=>id,1", 'Time,f=>time,key&n', "[PositionX|PositionY|PositionZ],f=>p,n", "Length,f=>length,combined&n", "Width,f=>width,combined&n", "Height,=>height,combined&n",
    "Yaw,f=>yaw,n", "Pitch,f=>pitch,n", "Roll,f=>roll,n", "[VX|VY|VZ],f=>v,n", "[AX|AY|AZ],f=>a,n",
    "Category,s=>category,combined&n", "Style,s=>style,combined&n", "Color,s=>color,combined&n", "Ego,b=>ego,1"
]

FIELD_DEF_MAIN = [
    "ID,s=>id,1", 'Time,f=>time,key&n', "[PositionX|PositionY|PositionZ],f=>p,n",
    "Yaw,f=>yaw,n", "Pitch,f=>pitch,n", "Roll,f=>roll,n", "[VX|VY|VZ],f=>v,n", "[AX|AY|AZ],f=>a,n"
]

FIELD_DEF_OBJ = [
    "ID,s=>id,1", 'Time,f=>time,key&n', "[LongitudinalOffset|LateralOffset|UpOffset],f=>p,n",
    "Length,f=>length,combined&n", "Width,f=>width,combined&n", "Height,=>height,combined&n",
    "Yaw,f=>yaw,n", "Pitch,f=>pitch,n", "Roll,f=>roll,n", "[VX|VY|VZ],f=>v,n", "[AX|AY|AZ],f=>a,n",
    "Category,s=>category,combined&n", "Style,s=>style,combined&n", "Color,s=>color,combined&n"
]

class RigidBodyState:
    def __init__(self):
        self.p = None

    def __getattr__(self, item):
        if item.find("__") == 0:    # ignore internal attr/methods, such as __reduce_ex__
            return super(RigidBodyState, self).__getattr__(item)
        else:
            return None

class Trajectory:
    def __init__(self):
        self.seq = collections.OrderedDict()
        self._cached_timestamps = None
        self._cached_positions = None
        self._cached_quats = None

        self._cached_pos_f = None
        self._cached_orientation_f = None

    def __getattr__(self, item):
        if item.find("__") == 0:    # ignore internal attr/methods, such as __reduce_ex__
            return super(Trajectory, self).__getattr__(item)
        else:
            return None

    def pose_by_time(self, t):
        if self._cached_timestamps is None:
            self._cached_timestamps = np.array(list(self.seq.keys()))
        if self._cached_positions is None:
            self._cached_positions = self.all_pos3d()
        if self._cached_quats is None:
            self._cached_quats = [None] * self._cached_timestamps.shape[0]
            for i, state in enumerate(self.seq.values()):
                self._cached_quats[i] = Quaternion(axis=[0.0, 0.0, 1.0], degrees=state.yaw)

        idx = bisect.bisect_left(self._cached_timestamps, t)
        t0 = self._cached_timestamps[idx-1]
        t1 = self._cached_timestamps[idx]
        a = (t - t0) / (t1 - t0)

        new_p = (1-a)*self._cached_positions[idx-1] + a*self._cached_positions[idx]
        new_yaw = math.degrees(Quaternion.slerp(self._cached_quats[idx-1], self._cached_quats[idx], a).yaw_pitch_roll[0])
        return new_p, new_yaw

    def pose_by_time_slow(self, t):
        if self._cached_timestamps is None:
            self._cached_timestamps = np.array(list(self.seq.keys()))
        if self._cached_pos_f is None:
            self._cached_pos_f = interp1d(self._cached_timestamps, self.all_pos3d(), axis=0, assume_sorted=True)
        if self._cached_orientation_f is None:
            eulers = np.empty((self._cached_timestamps.shape[0], 3))
            for i, state in enumerate(self.seq.values()):
                eulers[i, 0] = state.yaw
            rots = Rotation.from_euler("zyx", eulers, degrees=True)
            self._cached_orientation_f = Slerp(self._cached_timestamps, rots)

        new_p2 = self._cached_pos_f(t)
        new_yaw2 = self._cached_orientation_f(t).as_euler('zyx', degrees=True)[0]
        return new_p2, new_yaw2

    def timestamps(self):
        times = np.empty((len(self.seq.keys()), 1))
        for i, time in enumerate(self.seq.keys()):
            times[i] = time
        return times

    def all_pos2d(self):
        pos = np.empty((len(self.seq.keys()), 2))
        for i, time in enumerate(self.seq.keys()):
            c = self.seq[time].p
            pos[i,0] = c[0]
            pos[i,1] = c[1]
        return pos

    def all_pos3d(self):
        pos = np.empty((len(self.seq.keys()), 3))
        for i, time in enumerate(self.seq.keys()):
            c = self.seq[time].p
            pos[i,0] = c[0]
            pos[i,1] = c[1]
            pos[i,2] = 0.0 if c[2] is None else c[2]
        return pos

    def all_shape3d(self):
        vehicle_shape = np.empty((len(self.seq.keys()), 3))
        for i, time in enumerate(self.seq.keys()):
            vehicle_shape[i, :3] = (
                self.seq[time].length,
                self.seq[time].width,
                self.seq[time].height,
            )
        return vehicle_shape
    
    def set_attr(self, attr, new_value):
        for v in self.seq.values():
            setattr(v, attr, new_value)

    def set_all_pos2d(self, processed_pos):
        if len(self.seq) != len(processed_pos):
            print("Lengths of position arrays are mismatched!")
        for i, v in enumerate(self.seq.values()):
            v.p[0] = processed_pos[i][0]
            v.p[1] = processed_pos[i][1]


def _read_field(type, row, idx_range):
    idx, idx2 = idx_range
    if idx2 == idx + 1:
        if type == "s":
            return None if row[idx] is None or row[idx] == '' else row[idx]
        if type == "b":
            return None if row[idx] is None or row[idx] == '' else (row[idx].lower() == "y" or row[idx].lower() == "yes" or row[idx].lower() == "true")
        else:
            return float(row[idx]) if row[idx] != '' else None
    else:
        ret = []
        for i in range(idx, idx2):
            ret.append(float(row[i]) if row[i] != '' else None)
        return ret

class FieldDef:
    def __init__(self, def_string, idx):
        field, attr = def_string.split("=>")
        field_name, self.field_type = field.split(",")
        self.field_name = field_name.strip("[]")
        self.attr_name, attr_type = attr.split(",")
        self.attr_type = attr_type.split("&")
        self.elements = self.field_name.split("|")
        self.index_range = (idx, idx + len(self.elements))

class StandardRawDataHolder:
    def _parse_field_def(self, field_def_strings):
        d = collections.OrderedDict()
        element_count = 0
        for def_string in field_def_strings:
            f = FieldDef(def_string, element_count)
            d[f.field_name] = f
            element_count += len(f.elements)
        return d
        
    def __init__(self, setting=None):
        if setting is None:
            field_def_strings = FIELD_DEF
        elif isinstance(setting, str):
            if setting == "unified":
                field_def_strings = FIELD_DEF
            elif setting == "main":
                field_def_strings = FIELD_DEF_MAIN
            elif setting == "objs":
                field_def_strings = FIELD_DEF_OBJ
        else:   # string list
            field_def_strings = setting
        self.field_defs = self._parse_field_def(field_def_strings)
        self.objs = collections.OrderedDict()

    def redefine_fields(self, field_def_strings):
        self.field_defs = self._parse_field_def(field_def_strings)

    def read(self, file_name, file_type=None):
        with open(file_name, 'r') as csv_file:
            for row_num, row_str in enumerate(csv_file):
                if row_num != 0:
                    row = [v.strip() for v in row_str.split(',')]
                    state = RigidBodyState()
                    traj = None
                    for field_def in self.field_defs.values():
                        if field_def.attr_name == "id":
                            id = _read_field(field_def.field_type, row, field_def.index_range)
                            if id not in self.objs:
                                traj = Trajectory()
                                for field_def2 in self.field_defs.values():
                                    if "combined" in field_def2.attr_type:
                                        setattr(traj, field_def2.attr_name, None)
                                self.objs[id] = traj
                            traj = self.objs[id]
                        if "1" in field_def.attr_type:
                            setattr(traj, field_def.attr_name, _read_field(field_def.field_type, row, field_def.index_range))
                        if "n" in field_def.attr_type:
                            setattr(state, field_def.attr_name, _read_field(field_def.field_type, row, field_def.index_range))
                        if "key" in field_def.attr_type:
                            key = _read_field(field_def.field_type, row, field_def.index_range)
                            setattr(state, field_def.attr_name, key)
                    if key in traj.seq:
                        print("Found data with exactly the same timestamp: id=%s, time=%f" % (traj.id, key))
                    traj.seq[key] = state

        # extract per-trajectory attributes
        for _, traj in self.objs.items():
            count = 0
            for state in traj.seq.values():
                for field_def in self.field_defs.values():
                    if "combined" in field_def.attr_type:
                        v1 = getattr(state, field_def.attr_name)
                        v2 = getattr(traj, field_def.attr_name)
                        setattr(traj, field_def.attr_name, v1 if count == 0 or v1 == v2 else None)
                count += 1

    def csv_rows(self, traj):
        for state in traj.seq.values():
            row = {}
            for field_name, field_def in self.field_defs.items():
                if "combined" in field_def.attr_type:
                    v = getattr(traj, field_def.attr_name)
                    if v is None:
                        v = getattr(state, field_def.attr_name)
                    row[field_name] = v
                if "1" in field_def.attr_type:
                    v = getattr(traj, field_def.attr_name)
                    if v is not None:
                        if "b" in field_def.field_type:
                            row[field_name] = "Y" if v else "N"
                        else:
                            row[field_name] = v
                if "n" in field_def.attr_type:
                    if len(field_def.elements) > 1:
                        elements = getattr(state, field_def.attr_name)
                        for i, element_name in enumerate(field_def.elements):
                            row[element_name] = elements[i] if elements != None else None
                    else:
                        if field_def.field_name not in row: # don't override combined attribs from traj
                            row[field_def.field_name] = getattr(state, field_def.attr_name)
            yield row

    def write(self, output_file_path, append=False):
        with open(output_file_path, 'a' if append else "w", newline='') as out_file:
            field_names = []
            for field_name in self.field_defs.keys():
                field_names += field_name.split("|")

            writer = csv.DictWriter(out_file, fieldnames = field_names)
            if not append:
                writer.writeheader()

            for traj in self.objs.values():
                for row in self.csv_rows(traj):
                    writer.writerow(row)

    def set_all_pos2d(self, id, processed_pos):
        self.objs[id].set_all_pos2d(processed_pos)

    def set_attr(self, id, attr, new_value):
        self.objs[id].set_attr(attr, new_value)
        setattr(self.objs[id], attr, new_value)

    def copy_obj_metadata(self, id):
        new_traj = Trajectory()
        for field_def in self.field_defs.values():
            if "1" in field_def.attr_type or "combined" in field_def.attr_type:
                v = getattr(self.objs[id], field_def.attr_name)
                setattr(new_traj, field_def.attr_name, v)
        return new_traj

class StandardRawDataMeta:
    def read(self, path):
        meta = read_json(path)
        self.category = meta["category"]
        self.type = meta["type"]
        self.map = meta["map"]
        if meta["map_background"]:
            self.map_background = meta["map_background"]
            self.map_background_bbox = meta["map_background_bbox"]
        else:
            self.map_background = ''
        self.main_vehicle_id = meta["main_vehicle_id"]
        self.trajs = meta["trajectories"]
        if self.type == "relative_objs":
            self.sensor_settings = meta["sensor_settings"]


    def write(self, path):
        meta = {
            "category": self.category,
            "type": self.type,
            "map": self.map,
            "map_background": self.map_background,
            "main_vehicle_id": self.main_vehicle_id,
            "trajectories": self.trajs,
        }
        if self.type == "relative_objs":
            meta["sensor_settings"] = self.sensor_settings
        if self.map_background:
            meta["map_background_bbox"] = self.map_background_bbox

        write_json(path, meta)

class StandardRawData:
    def __init__(self):
        self.meta = StandardRawDataMeta()
        self.trajs = []

    def read(self, path):
        self.meta.read(path)
        self.base_path = os.path.split(path)[0]
        if self.meta.type == "unified":
            holder = StandardRawDataHolder(FIELD_DEF)
            path1 = os.path.join(self.base_path, self.meta.trajs)
            holder.read(path1)
            self.trajs.append(holder)
        else: # if self.meta.type == "relative_objs":
            main_holder = StandardRawDataHolder(FIELD_DEF_MAIN)
            path1 = os.path.join(self.base_path, self.meta.trajs["main"])
            main_holder.read(path1)
            self.trajs.append(main_holder)
            objs_holder = StandardRawDataHolder(FIELD_DEF_OBJ)
            path2 = os.path.join(self.base_path, self.meta.trajs["objs"])
            objs_holder.read(path2)
            self.trajs.append(objs_holder)

    def write(self, path, type=None):
        base_path, meta_name = os.path.split(path)
        if self.meta.type == "unified" and (type is None or type == self.meta.type):
            unified_csv_name = meta_name.replace(".meta.json", ".csv")
            unified_csv_path = os.path.join(base_path, unified_csv_name)
            self.trajs[0].write(unified_csv_name)
            self.meta.write(path)
        elif self.meta.type == "relative_objs" and (type is None or type == self.meta.type):
            main_csv_name = meta_name.replace(".meta.json", ".main.csv")
            main_csv_path = os.path.join(base_path, main_csv_name)
            self.trajs[0].write(main_csv_path)  
            objs_csv_name = meta_name.replace(".meta.json", ".objs.csv")
            objs_csv_path = os.path.join(base_path, objs_csv_name)
            self.trajs[1].write(objs_csv_path)
            self.meta.trajs = {'main': main_csv_name, 'objs': objs_csv_name}
            self.meta.write(path)
        elif type == "unified" and self.meta.type == "relative_objs":
            unified_csv_name = meta_name.replace(".meta.json", ".csv")
            unified_csv_path = os.path.join(base_path, unified_csv_name)
            self.trajs[0].redefine_fields(FIELD_DEF)
            self.trajs[0].write(unified_csv_path)
            self.trajs[1].redefine_fields(FIELD_DEF)
            self.trajs[1].write(unified_csv_path, append=True)
            self.meta.trajs = unified_csv_name
            self.meta.type = type
            self.meta.write(path)

if __name__ == '__main__':
    raw_data_holder = StandardRawDataHolder()
    raw_data_holder.read("test/64_20200529172236_offset.csv")
    raw_data_holder.write("test/64_20200529172236_offset.out.csv")

    raw_data_holder = StandardRawDataHolder(FIELD_DEF_MAIN)
    raw_data_holder.read("test/20200714_test01.main.csv")
    raw_data_holder.write("test/20200714_test01.main.out.csv")

    raw_data_holder = StandardRawDataHolder(FIELD_DEF_OBJ)
    raw_data_holder.read("test/20200714_test01.objs.csv")
    raw_data_holder.write("test/20200714_test01.objs.out.csv")
