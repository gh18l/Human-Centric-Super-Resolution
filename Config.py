import scipy.io as sio
import numpy as np
import os
import pickle
import json
import cv2
import csv

render_fingers = True
render_toes = True
render_hands = True
HEAD_VID = 411
SMPL_COCO_PATH = 'Data/Smpl_Model/neutral_smpl_with_cocoplus_reg.pkl'
SMPL_NORMAL_PATH = 'Data/Smpl_Model/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
POSE_PRIOR_PATH = 'Data/Prior/genericPrior.mat'
hmr_path = "/home/guanghan/real_system_data7/data1/people3/HR/"
texture_path = "/home/guanghan/real_system_data7/data1/people3/HR/output/texture_file/"
HR_pose_path = "/home/guanghan/real_system_data7/data1/people3/HR/output/"
refine_output_path = "output_after_refine_prior3000"
output_path = "output_after_refine_prior3000"
refine_reference_path = "output_after_refine_prior3000"

openpose_path = "/home/lgh/Downloads/openpose"
PSP_path = "/home/lgh/Downloads/PSPNet-Keras-tensorflow-master"
hmr_path = "/home/lgh/Downloads/hmr-master"

video = False
pedestrian_constraint = True
mode = "pose"  ##### pose or full
img_widthheight = 115002200
img_width = 1500
img_height = 2200
graphcut_index = 30
SMPL_JOINT_IDS = [11, 10, 8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12]

J2D_Loss = 1.0
J2D_face_Loss = 2.0
J2D_head_Loss = 1.0
J2D_foot_Loss = 1.0
mask = 0.05
hmr_constraint = 6000.0
hmr_hands_constraint = 0.0
temporal3d = 1000.0
 temporal2d = 1000.0
temporal_pose = 1000.0
dense_optflow = 0.08

J2D_refine_Loss = 1.0
J2D_face_refine_Loss = 2.0
J2D_head_refine_Loss = 1.0
J2D_foot_refine_Loss = 1.0
hmr_constraint_refine = 6000.0
temporal3d_refine = 1000.0
temporal2d_refine = 1000.0
temporal_pose_refine = 1000.0
match_idx = 485

batch_num = 1


    def load_initial_param(self):
        pose_prior = sio.loadmat(self.POSE_PRIOR_PATH, squeeze_me=True, struct_as_record=False)
        pose_mean = pose_prior['mean']
        pose_covariance = np.linalg.inv(pose_prior['covariance'])
        zero_shape = np.ones([13]) * 1e-8  # extra 3 for zero global rotation
        zero_trans = np.ones([3]) * 1e-8
        initial_param = np.concatenate([zero_shape, pose_mean, zero_trans], axis=0)

        return initial_param, pose_mean, pose_covariance

    def load_pose_pkl(self):
        if os.path.exists(self.hmr_path + self.output_path):
            pose_path = self.hmr_path + self.output_path
            pose_pkl_files = os.listdir(pose_path)
            pose_pkl_files = sorted([filename for filename in pose_pkl_files if filename.endswith(".pkl")],
                                  key=lambda d: int((d.split('_')[3]).split('.')[0]))
            j3dss = []
            for ind, pose_pkl_file in enumerate(pose_pkl_files):
                HR_pkl_path = os.path.join(pose_path, pose_pkl_file)
                with open(HR_pkl_path) as f:
                    param = pickle.load(f)
                j3ds = param['j3ds']
                j3dss.append(j3ds)
            return j3dss, True
        else:
            return [], False

    def load_camera_pkl(self):
        if os.path.exists(self.hmr_path + self.output_path):
            path = self.hmr_path + self.output_path
            pkl_files = os.listdir(path)
            pkl_files = sorted([filename for filename in pkl_files if filename.endswith(".pkl")],
                                    key=lambda d: int((d.split('_')[3]).split('.')[0]))
            cams = []
            for ind, pkl_file in enumerate(pkl_files):
                pkl_path = os.path.join(path, pkl_file)
                with open(pkl_path) as f:
                    param = pickle.load(f)
                cam = param['cam']
                cams.append(cam)
            return cams, True
        else:
            return [], False

    def load_hmr_data(self):
        hmr_theta, hmr_beta, hmr_tran, hmr_cam, hmr_joint3d = get_hmr(self.hmr_path)
        hmr_dict = {"hmr_thetas": hmr_theta, "hmr_betas": hmr_beta,
                    "hmr_trans": hmr_tran, "hmr_cams": hmr_cam,
                    "hmr_joint3ds": hmr_joint3d}
        path = self.hmr_path + "/optimization_data"
        COCO_path = path + "/COCO"
        MPI_path = path + "/MPI"
        imgs = []
        j2ds = []
        confs = []
        j2ds_head = []
        confs_head = []
        j2ds_face = []
        confs_face = []
        j2ds_foot = []
        confs_foot = []
        masks = []

        COCO_j2d_files = os.listdir(COCO_path)
        COCO_j2d_files = sorted([filename for filename in COCO_j2d_files if filename.endswith(".json")],
                                key=lambda d: int((d.split('_')[0])))
        MPI_j2d_files = os.listdir(MPI_path)
        MPI_j2d_files = sorted([filename for filename in MPI_j2d_files if filename.endswith(".json")],
                               key=lambda d: int((d.split('_')[0])))
        img_files = os.listdir(path)
        img_files = sorted([filename for filename in img_files if (filename.endswith(".png") or filename.endswith(
            ".jpg")) and "mask" not in filename and "label" not in filename and "std" not in filename])
        # key=lambda d: int((d.split('_')[0])))

        mask_files = os.listdir(path)
        mask_files = sorted([filename for filename in mask_files if filename.endswith(".png") and "mask" in filename],
                            key=lambda d: int((d.split('_')[1]).split('.')[0]))

        for ind, COCO_j2d_file in enumerate(COCO_j2d_files):
            coco_j2d_file_path = os.path.join(COCO_path, COCO_j2d_file)
            coco_j2d, coco_conf = load_openposeCOCO(coco_j2d_file_path)
            mpi_j2d_file_path = os.path.join(MPI_path, MPI_j2d_files[ind])
            mpi_j2d, mpi_conf = load_openposeCOCO(mpi_j2d_file_path)
            j2ds.append(coco_j2d[0:14, :])
            confs.append(coco_conf[0:14])
            j2ds_face.append(coco_j2d[14:19, :])
            confs_face.append(coco_conf[14:19])
            j2ds_head.append(mpi_j2d[1::-1, :])
            confs_head.append(mpi_conf[1::-1])

            ## deal with foot joints
            _confs_foot = np.zeros(2)
            if coco_conf[19] != 0 and coco_conf[20] != 0:
                _confs_foot[0] = (coco_conf[19] + coco_conf[20]) / 2.0
            else:
                _confs_foot[0] = 0.0
            if coco_conf[22] != 0 and coco_conf[23] != 0:
                _confs_foot[1] = (coco_conf[22] + coco_conf[23]) / 2.0
            else:
                _confs_foot[1] = 0.0
            _j2ds_foot = np.zeros([2, 2])
            _j2ds_foot[0, 0] = (coco_j2d[19, 0] + coco_j2d[20, 0]) / 2.0
            _j2ds_foot[0, 1] = (coco_j2d[19, 1] + coco_j2d[20, 1]) / 2.0
            _j2ds_foot[1, 0] = (coco_j2d[22, 0] + coco_j2d[23, 0]) / 2.0
            _j2ds_foot[1, 1] = (coco_j2d[22, 1] + coco_j2d[23, 1]) / 2.0

            j2ds_foot.append(_j2ds_foot)
            confs_foot.append(_confs_foot)

            img_file_path = os.path.join(path, img_files[ind])
            img = cv2.imread(img_file_path)
            imgs.append(img)

            mask_file_path = os.path.join(path, mask_files[ind])
            mask1 = cv2.imread(mask_file_path)
            ##################cautious change!!!!!!!!!!!################
            mask = mask1[:, :, 0]
            mask[mask < 255] = 0
            masks.append(mask)
        data_dict = {"j2ds": np.array(j2ds), "confs": np.array(confs), "imgs": imgs,
                     "masks": masks, "j2ds_face": np.array(j2ds_face),
                     "confs_face": np.array(confs_face), "j2ds_head": np.array(j2ds_head),
                     "confs_head": np.array(confs_head), "j2ds_foot": np.array(j2ds_foot),
                     "confs_foot": np.array(confs_foot)}
        return hmr_dict, data_dict

def get_original(proc_param, cam):
    img_size = proc_param['img_size']
    undo_scale = 1. / np.array(proc_param['scale'])

    cam_s = cam[0]
    cam_pos = cam[1:]
    flength = 500.
    tz = flength / (0.5 * img_size * cam_s)
    trans = np.hstack([cam_pos, tz])
    return trans

## trans3 pose72 beta10 =85
def get_hmr(hmr_init_path):
    hmr_init_files = os.listdir(hmr_init_path)
    hmr_init_files = sorted([filename for filename in hmr_init_files
                        if filename.endswith(".npy") and "theta" in filename],
                            key=lambda d: int((d.split('_')[1]).split('.')[0]))

    N = len(hmr_init_files)
    print("%d files" % N)
    hmr_tran = np.zeros([N, 3])
    hmr_theta = np.zeros([N, 72])
    hmr_beta = np.zeros([N, 10])
    hmr_cam = np.zeros([N, 3])
    hmr_joints3d = np.zeros([N, 19, 3])

    hmrcam_init_files = os.listdir(hmr_init_path)
    hmrcam_init_files = sorted([filename for filename in hmrcam_init_files
                             if filename.endswith(".npy") and "camera" in filename],
                            key=lambda d: int((d.split('_')[3]).split('.')[0]))
    hmrproc_init_files = os.listdir(hmr_init_path)
    hmrproc_init_files = sorted([filename for filename in hmrproc_init_files
                                if filename.endswith(".npy") and "proc" in filename],
                               key=lambda d: int((d.split('_')[2]).split('.')[0]))
    hmrjoints3d_init_files = os.listdir(hmr_init_path)
    hmrjoints3d_init_files = sorted([filename for filename in hmrjoints3d_init_files
                                 if filename.endswith(".npy") and "joints3d" in filename],
                                key=lambda d: int((d.split('_')[1]).split('.')[0]))
    for ind, hmr_init_file in enumerate(hmr_init_files):
        file = np.load(hmr_init_path + hmr_init_files[ind], allow_pickle=True)
        file_cam = np.load(hmr_init_path + hmrcam_init_files[ind], allow_pickle=True)
        file_proc = np.load(hmr_init_path + hmrproc_init_files[ind], allow_pickle=True).item()
        file_joints3d = np.load(hmr_init_path + hmrjoints3d_init_files[ind], allow_pickle=True)
        trans = get_original(file_proc, file[0, 0:3])

        hmr_tran[ind, :] = trans
        hmr_theta[ind, :] = file[0, 3:75]
        hmr_beta[ind, :] = file[0, 75:85]
        hmr_cam[ind, :] = file_cam
        hmr_joints3d[ind, :, :] = file_joints3d.squeeze()
    return hmr_theta, hmr_beta, hmr_tran, hmr_cam, hmr_joints3d

def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        kps.append(kp)
    return kps

def load_openposeCOCO(file):
    #file = "/home/lgh/Documents/2d_3d_con/aa_000000000000_keypoints.json"
    openpose_index = np.array([11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 1, 0, 0, 16, 15, 18, 17, 22, 23, 24, 20, 19, 21])
    deepcut_index = np.array([14, 13, 9, 8, 7, 10, 11, 12, 3, 2, 1, 4, 5, 6]) - 1 #openpose index 123456...

    kps = read_json(file)
    if len(kps) == 0:
        joints = np.zeros([50, 2])
        conf = np.zeros(50)
        return joints, conf

    scores = [np.mean(kp[kp[:, 2] > -0.1, 2]) for kp in kps]

    _data = kps[np.argmax(scores)].reshape(-1)
    #_data = kps[0].reshape(-1)
    joints = []
    conf = []

    if len(_data) == 45:
        for o in range(0, len(_data), 3):
            temp = []
            temp.append(_data[o])
            temp.append(_data[o + 1])
            joints = np.vstack((joints, temp)) if len(joints) != 0 else temp
            conf = np.vstack((conf, _data[o + 2])) if len(conf) != 0 else np.array([_data[o + 2]])
        conf = conf.reshape((len(conf,)))
        return (joints, conf)

    deepcut_joints = []
    deepcut_conf = []
    for o in range(0, len(_data), 3):   #15 16 17 18 is miss
        temp = []
        temp.append(_data[o])
        temp.append(_data[o + 1])
        joints = np.vstack((joints, temp)) if len(joints) != 0 else temp
        conf = np.vstack((conf, _data[o + 2])) if len(conf) != 0 else np.array([_data[o + 2]])

    #demo_point(joints[:,0], joints[:,1])

    for o in range(14+5+6):
        deepcut_joints = np.vstack((deepcut_joints, joints[openpose_index[o]])) if len(deepcut_joints)!=0 else joints[openpose_index[o]]
        deepcut_conf = np.vstack((deepcut_conf, conf[openpose_index[o]])) if len(deepcut_conf)!=0 else np.array(conf[openpose_index[o]])
    deepcut_conf = deepcut_conf.reshape((len(deepcut_conf)))
    return deepcut_joints, deepcut_conf

def save_pkl_to_npy(pose_path):
    #####save csv before refine, extra output
    pkl_files = os.listdir(pose_path)
    pkl_files = sorted([filename for filename in pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int((d.split('_')[3]).split('.')[0]))
    length = len(pkl_files)
    array = np.zeros((length, 24 * 3))
    for ind, pkl_file in enumerate(pkl_files):
        pkl_path = os.path.join(pose_path, pkl_file)
        with open(pkl_path) as f:
            param = pickle.load(f)
        pose = param['pose']
        for i in range(24 * 3):
            array[ind, i] = pose.squeeze()[i]
    np.save(os.path.join(pose_path, "optimization_pose.npy"), array)


def save_pkl_to_csv(pose_path):
    #####save csv before refine, extra output
    pkl_files = os.listdir(pose_path)
    pkl_files = sorted([filename for filename in pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int((d.split('_')[3]).split('.')[0]))
    length = len(pkl_files)
    array = np.zeros((length, 24 * 3))
    for ind, pkl_file in enumerate(pkl_files):
        pkl_path = os.path.join(pose_path, pkl_file)
        with open(pkl_path) as f:
            param = pickle.load(f)
        pose = param['pose']
        for i in range(24 * 3):
            array[ind, i] = pose.squeeze()[i]
    with open(os.path.join(pose_path, "optimization_pose.csv"), "w") as f:
        writer = csv.writer(f)
        for row in array:
            writer.writerow(row)
