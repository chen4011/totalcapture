import os
from collections import deque
import re
import numpy as np
from pyquaternion import Quaternion
import bvh


bone_names = ('Head', 'Sternum', 'Pelvis', 'L_UpArm', 'R_UpArm', 'L_LowArm', 'R_LowArm',
              'L_UpLeg', 'R_UpLeg', 'L_LowLeg', 'R_LowLeg', 'L_Foot', 'R_Foot')  # imu bones

# marker name of vicon
vicon_joints = ('Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head', 'RightShoulder',
                'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm',
                'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot')

bvh_joints = ('Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
              'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandEnd', 'RightHandThumb1',
              'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandEnd', 'LeftHandThumb1',
              'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
              'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase')

# should check bone start and end order, e.g. neck to head or vice versa
imu_bone_vicon_start = {'Head': 'Neck', 'Sternum': 'Spine3', 'Pelvis': 'Spine',  # not sure about this row
                        'L_UpArm': 'LeftArm', 'R_UpArm': 'RightArm',
                        'L_LowArm': 'LeftForeArm', 'R_LowArm': 'RightForeArm',
                        'L_UpLeg': 'LeftUpLeg', 'R_UpLeg': 'RightUpLeg',
                        'L_LowLeg': 'LeftLeg', 'R_LowLeg': 'RightLeg',
                        'L_Foot': 'LeftFoot', 'R_Foot': 'RightFoot'}

imu_bone_vicon_end   = {'Head': 'Head', 'Sternum': 'Neck', 'Pelvis': 'Spine1',  # not sure about this row
                        'L_UpArm': 'LeftForeArm', 'R_UpArm': 'RightForeArm',
                        'L_LowArm': 'LeftHand', 'R_LowArm': 'RightHand',
                        'L_UpLeg': 'LeftLeg', 'R_UpLeg': 'RightLeg',
                        'L_LowLeg': 'LeftFoot', 'R_LowLeg': 'RightFoot',
                        'L_Foot': 'LeftToeBase', 'R_Foot': 'RightToeBase'}

# re
float_reg = r'(\-?\d+\.\d+)'
int_reg = r'[\d]+'
word_reg = r'[a-zA-z\_]+'

float_finder = re.compile(float_reg)
int_finder = re.compile(int_reg)
word_finder = re.compile(word_reg)


def parse_sensor_6axis(fpath):
    # 開啟給定的文件路徑fpath，並讀取所有行到lines列表中
    with open(fpath, 'r') as f:
        lines = f.readlines()
    
    # 從lines列表中彈出第一行，並從中解析出感測器數量和幀數
    lines = deque(lines)
    seq_meta = lines.popleft()  # # 從lines deque中移除並返回第一行文本，並將其存儲在seq_meta變量中；popleft()方法在deque不為空的情況下，會從左側移除並返回一個元素
    seq_meta_result = int_finder.findall(seq_meta)  # 使用正則表達式int_finder在seq_meta文本中查找所有的整數，並將結果存儲在seq_meta_result列表中
    assert len(seq_meta_result) == 2, 'error in seq meta data'  # 確保seq_meta_result列表的長度為2，否則會引發異常。這表示seq_meta文本應該包含兩個整數
    # 從seq_meta_result列表中提取出兩個整數，並將它們轉換為Python的int類型，然後分別存儲在num_sensors和num_frames變量中
    num_sensors = int(seq_meta_result[0])   # 感測器數量，13
    num_frames = int(seq_meta_result[1])    # 幀數量，4115

    # 初始化一個空列表frames來存儲每一幀的感測器數據
    frames = []
    # 對於每一幀，進行以下操作
    for f in range(num_frames):
        # frame
        # 從lines deque中彈出一行文本，並從中解析出幀索引
        frame_index_line = lines.popleft()
        frame_index = int(int_finder.findall(frame_index_line)[0])

        # 初始化一個空字典joints來存儲該幀的感測器數據，並將幀索引添加到joints字典中
        joints = dict()
        joints['index'] = frame_index  # frame index
        # joint

        # 對於每一個感測器，進行以下操作
        for i in range(num_sensors):
            # 從lines deque中彈出一行文本，並從中解析出感測器名稱和其對應的數據
            onejoint = lines.popleft()

            # 確保感測器名稱在bone_names列表中
            joint_name = word_finder.findall(onejoint)[0]
            assert joint_name in bone_names, 'invalid joint name: {} in frame {}'.format(joint_name, frame_index)

            # 確保數據的長度為7（因為每個感測器的數據由四元數的方向和三維的加速度組成）
            values = float_finder.findall(onejoint)
            assert len(values) == 7, 'wrong number of joint parameter'

            # 將方向和加速度數據轉換為浮點數，並存儲在orientation和acceleration變量中
            orientation = tuple(float(x) for x in values[0:4])
            acceleration = tuple(float(x) for x in values[4:7])
            joints[joint_name] = (orientation, acceleration)

        frames.append(joints)
    return frames


def parse_calib_imu_ref(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    lines = deque(lines)
    ref_num_sensors = int(lines.popleft())
    assert ref_num_sensors == len(bone_names), 'mismatching sensor nums with ref sensor nums'
    ref_joints = dict()
    for i in range(ref_num_sensors):
        onejoint = lines.popleft()
        joint_name = word_finder.findall(onejoint)[0]
        assert joint_name in bone_names, 'invalid joint name: {}'.format(joint_name)
        values = float_finder.findall(onejoint)
        assert len(values) == 4, 'wrong number of joint parameter'
        # orientation in ref is ordered as (x y z w), which is different from captured data (w x y z)
        orientation_imag = [float(x) for x in values[0:3]]
        orientation = [float(values[3])]
        orientation.extend(orientation_imag)
        ref_joints[joint_name] = orientation
    return ref_joints


def parse_calib_imu_bone(fpath):
    # 開啟給定的文件路徑fpath，並讀取所有行到lines列表中
    with open(fpath, 'r') as f:
        lines = f.readlines()

    # 將lines列表轉換為一個deque（雙端隊列）
    lines = deque(lines)

    # 從lines deque中移除並返回第一行，並將其轉換為整數，然後存儲在bone_num_sensors變量中。這個變量表示感測器的數量，即 13
    bone_num_sensors = int(lines.popleft())

    # 確保bone_num_sensors等於bone_names列表的長度，否則會引發異常，表示感測器的數量應該與骨骼的數量相等
    assert bone_num_sensors == len(bone_names), 'mismatching sensor nums with ref sensor nums'

    # 初始化一個空字典ref_bones來存儲每個骨骼的校準數據
    ref_bones = dict()
    # 對於每一個感測器，進行以下操作
    for i in range(bone_num_sensors):
        # 從lines deque中移除並返回一行，並將其存儲在onejoint變量中
        onejoint = lines.popleft()

        # 從onejoint文本中查找第一個單詞，並將其存儲在joint_name變量中。這個變量表示關節的名稱
        joint_name = word_finder.findall(onejoint)[0]

        # 確保joint_name在bone_names列表中，否則會引發異常
        assert joint_name in bone_names, 'invalid joint name: {}'.format(joint_name)

        # 從onejoint文本中查找所有的浮點數，並將結果存儲在values列表中
        values = float_finder.findall(onejoint)

        # 確保values列表的長度為4，否則會引發異常。這表示每個關節應該有四個校準參數
        assert len(values) == 4, 'wrong number of joint parameter'

        # orientation in ref is ordered as (x y z w), which is different from captured data (w x y z)
        # 從values列表中提取出四元數的值，並將其順序調整為[w, x, y, z]，然後存儲在orientation列表中
        orientation_imag = [float(x) for x in values[0:3]]
        orientation = [float(values[3])]
        orientation.extend(orientation_imag)

        # 將joint_name和orientation添加到ref_bones字典中
        ref_bones[joint_name] = orientation
    return ref_bones


def parse_vicon_gt_ori(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    joints = lines[0].split()
    assert joints == list(vicon_joints), 'mismatching joint names with vicon gt'

    ori_frames = []
    for i in range(1,len(lines)):
        oneline = lines[i]
        vals = list(map(float, oneline.split()))
        if len(vals) == 0:
            break  # in case empty line at the file end

        # 確保vals列表的長度等於關節數量的四倍（因為每個關節的方向由四個數字的四元數表示）
        assert len(vals) == 4*len(joints), 'oops, mismatching joint nums and orientation data'
        joint_ori_frame = dict()
        for j in range(len(joints)):
            # quaternion order: xyzw, should do order manipulation
            joint_ori_frame[joints[j]] = [vals[4*j+3], vals[4*j], vals[4*j+1], vals[4*j+2]]
        ori_frames.append(joint_ori_frame)
    return ori_frames


def parse_vicon_gt_pos(fpath):
    # 開啟給定的文件路徑fpath，並讀取所有行到lines列表中
    with open(fpath, 'r') as f:
        lines = f.readlines()
    
    # 從第一行解析出關節名稱，並確保它們與vicon_joints列表中的關節名稱相匹配
    joints = lines[0].split()
    assert joints == list(vicon_joints), 'mismatching joint names with vicon gt'
    
    # 初始化一個空列表pos_frames來存儲每一幀的關節位置數據
    pos_frames = []
    
    # 對於lines中的每一行（除了第一行），進行以下操作
    for i in range(1,len(lines)):
        # 將該行的數據分割並轉換為浮點數，並存儲在vals列表中
        oneline = lines[i]
        vals = list(map(float, oneline.split()))

        # 如果vals列表為空，則跳出循環
        if len(vals) == 0:
            break  # in case empty line at the file end

        # 確保vals列表的長度等於關節數量的三倍（因為每個關節有三個坐標：x、y和z）
        assert len(vals) == 3*len(joints), 'oops, mismatching joint nums and position data'
        
        # 初始化一個空字典joint_pos_frame來存儲該幀的關節位置數據
        joint_pos_frame = dict()

        # 對於每一個關節，將其對應的三個坐標值添加到joint_pos_frame字典中
        for j in range(len(joints)):
            joint_pos_frame[joints[j]] = vals[3*j:3*(j+1)]

        # 將joint_pos_frame字典添加到pos_frames列表中
        pos_frames.append(joint_pos_frame)
    return pos_frames


def parse_imu_bone_info(fpath):
    # 開啟給定的文件路徑fpath，並讀取所有內容
    with open(fpath, 'r') as f:
        # 使用bvh.Bvh將讀取的內容解析為BVH（BioVision Hierarchy）動作捕捉數據
        mocap = bvh.Bvh(f.read())

    # 從BVH數據中獲取所有的關節，並將其存儲在all_joints變量中
    all_joints = mocap.get_joints()
    # 從BVH數據中獲取所有關節的名稱，並將其存儲在joints_names變量中
    joints_names = mocap.get_joints_names()

    # 初始化一個空字典bone_info來存儲每個骨骼的信息
    bone_info = dict()
    # 對於每一個骨骼，進行以下操作
    for b in bone_names:
        # 從imu_bone_vicon_start字典中獲取該骨骼的起始關節，並將其存儲在start變量中
        start = imu_bone_vicon_start[b]

        # 從imu_bone_vicon_end字典中獲取該骨骼的結束關節，並將其存儲在end變量中
        end = imu_bone_vicon_end[b]

        # 如果起始關節等於結束關節，則從all_joints中獲取該關節的索引，並從其子節點中獲取骨骼的長度，然後將其存儲在bone_length變量中
        if start == end:
            this_joint = all_joints[mocap.get_joint_index(start)]
            bone_length = tuple(float(x) for x in this_joint.children[2].children[0].value[1:])
        # 如果起始關節與結束關節不同，則從BVH數據中獲取結束關節的偏移量，並將其存儲在bone_length變量中
        else:
            bone_length = mocap.joint_offset(end)

        # 將骨骼的起始關節、結束關節和長度添加到bone_info字典中
        bone_info[b] = (start, end, bone_length)
    return bone_info


def parse_camera_cal(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    lines = deque(lines)
    num_cameras, distortion_order = lines.popleft().split()
    num_cameras = int(num_cameras)
    distortion_order = int(distortion_order)
    cameras = []
    for i in range(num_cameras):
        min_row, max_row, min_col, max_col = tuple(map(int, lines.popleft().split()))
        fx, fy, cx, cy = tuple(map(float, lines.popleft().split()))
        distor_param = float(lines.popleft())
        r1 = list(map(float, lines.popleft().split()))
        r2 = list(map(float, lines.popleft().split()))
        r3 = list(map(float, lines.popleft().split()))
        R = np.array([r1, r2, r3])
        t = np.reshape(np.array(list(map(float, lines.popleft().split()))), (3,1))
        cam = {'R': R, 'T': t, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'distor': distor_param}
        cameras.append(cam)
    return cameras
