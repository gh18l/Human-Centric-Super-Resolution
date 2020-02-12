import Utility as util_func
from Utility import Utility
import os
import numpy as np
import pickle
import algorithms
import cv2
from smpl_batch import SMPL
import tensorflow as tf
from camera import Perspective_Camera
import time
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
from opendr_render import render
import smpl_np
import pickle as pkl
import json

def refine_optimization(poses, betas, trans, data_dict, hmr_dict, LR_cameras, texture_img, texture_vt, Util):
    frame_num = len(poses)
    start_time = time.time()
    start_time_total = time.time()
    j3dss = Util.load_pose_pkl()
    j2ds = data_dict["j2ds"][:frame_num,:,:]
    confs = data_dict["confs"][:frame_num,:]
    j2ds_face = data_dict["j2ds_face"][:frame_num,:,:]
    confs_face = data_dict["confs_face"][:frame_num,:]
    j2ds_head = data_dict["j2ds_head"][:frame_num,:,:]
    confs_head = data_dict["confs_head"][:frame_num,:]
    j2ds_foot = data_dict["j2ds_foot"][:frame_num,:,:]
    confs_foot = data_dict["confs_foot"][:frame_num,:]
    imgs = data_dict["imgs"][:frame_num]

    Util.img_width = imgs[0].shape[1]
    Util.img_height = imgs[0].shape[0]
    Util.img_widthheight = int("1" + "%04d" % Util.img_width + "%04d" % Util.img_height)

    hmr_thetas = hmr_dict["hmr_thetas"][:frame_num,:]
    hmr_betas = hmr_dict["hmr_betas"][:frame_num,:]
    hmr_trans = hmr_dict["hmr_trans"][:frame_num,:]
    hmr_cams = hmr_dict["hmr_cams"][:frame_num,:]
    hmr_joint3ds = hmr_dict["hmr_joint3ds"][:frame_num,:,:]

    smpl_model = SMPL(Util.SMPL_COCO_PATH, Util.SMPL_NORMAL_PATH)

    initial_param, pose_mean, pose_covariance = Util.load_initial_param()

    param_shapes = tf.Variable(betas.reshape([-1, 10])[:frame_num,:], dtype=tf.float32)
    param_rots = tf.Variable(poses[:frame_num, :3].reshape([-1, 3])[:frame_num,:], dtype=tf.float32)
    param_poses = tf.Variable(poses[:frame_num, 3:72].reshape([-1, 69])[:frame_num,:], dtype=tf.float32)
    param_trans = tf.Variable(trans[:frame_num,:].reshape([-1, 3])[:frame_num,:], dtype=tf.float32)
    initial_param_tf = tf.concat([param_shapes, param_rots, param_poses, param_trans], axis=1)  ## N * (72+10+3)

    cam = Perspective_Camera(LR_cameras[0][0], LR_cameras[0][0], LR_cameras[0][1],
                             LR_cameras[0][2], np.zeros(3), np.zeros(3))
    j3ds, v, j3dsplus = smpl_model.get_3d_joints(initial_param_tf, Util.SMPL_JOINT_IDS)

    #### divide into different body parts
    j3ds_body = j3ds[:, 2:, :]
    j3ds_head = j3ds[:, 14:16, :]
    j3ds_foot = j3ds[:, :2, :]
    j3ds_face = j3dsplus[:, 14:19, :]
    j3ds_body = tf.reshape(j3ds_body, [-1, 3])  ## (N*12) * 3
    j3ds_head = tf.reshape(j3ds_head, [-1, 3])  ## (N*2) * 3
    j3ds_foot = tf.reshape(j3ds_foot, [-1, 3])  ## (N*2) * 3
    j3ds_face = tf.reshape(j3ds_face, [-1, 3])  ## (N*5) * 3
    j2ds_body_est = cam.project(tf.squeeze(j3ds_body))  ## (N*14) * 2
    j2ds_head_est = cam.project(tf.squeeze(j3ds_head))  ## (N*2) * 2
    j2ds_foot_est = cam.project(tf.squeeze(j3ds_foot))  ## (N*2) * 2
    j2ds_face_est = cam.project(tf.squeeze(j3ds_face))  ## (N*5) * 2

    v = tf.reshape(v, [-1, 3])  ## (N*6890) * 3
    verts_est_mask = cam.project(tf.squeeze(v))  ## (N*6890) * 2
    verts_est = cam.project(tf.squeeze(v))  ## (N*6890) * 2

    # TODO convert the loss function into batch input
    objs = {}

    j2ds = j2ds.reshape([-1, 2])  ## (N*14) * 2
    confs = confs.reshape(-1)  ## N*14
    base_weights = np.array(
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    base_weights = np.tile(base_weights, frame_num)  ## N*14
    weights = confs * base_weights  ## N*14
    weights = tf.constant(weights, dtype=tf.float32)  ## N*14
    objs['J2D_Loss'] = Util.J2D_refine_Loss * tf.reduce_sum(weights * tf.reduce_sum(tf.square(j2ds_body_est - j2ds), 1))

    j2ds_face = j2ds_face.reshape([-1, 2]) ## (N*5) * 2
    confs_face = confs_face.reshape(-1)  ## N*5
    base_weights_face = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0])
    base_weights_face = np.tile(base_weights_face, frame_num)  ## N*5
    weights_face = confs_face * base_weights_face
    weights_face = tf.constant(weights_face, dtype=tf.float32)
    objs['J2D_face_Loss'] = Util.J2D_face_refine_Loss * tf.reduce_sum(
        weights_face * tf.reduce_sum(tf.square(j2ds_face_est - j2ds_face), 1))

    j2ds_head = j2ds_head.reshape([-1, 2])  ## (N*2) * 2
    confs_head = confs_head.reshape(-1)  ## N*2
    base_weights_head = np.array(
        [1.0, 1.0])
    base_weights_head = np.tile(base_weights_head, frame_num)  ## N*2
    weights_head = confs_head * base_weights_head
    weights_head = tf.constant(weights_head, dtype=tf.float32)
    objs['J2D_head_Loss'] = Util.J2D_head_refine_Loss * tf.reduce_sum(
        weights_head * tf.reduce_sum(tf.square(j2ds_head - j2ds_head_est), 1))

    j2ds_foot = j2ds_foot.reshape([-1, 2])  ## (N*2) * 2
    confs_foot = confs_foot.reshape(-1) ## N*2
    base_weights_foot = np.array(
        [1.0, 1.0])
    base_weights_foot = np.tile(base_weights_foot, frame_num)  ## N*2
    weights_foot = confs_foot * base_weights_foot  ## N*2
    weights_foot = tf.constant(weights_foot, dtype=tf.float32)
    objs['J2D_foot_Loss'] = Util.J2D_foot_refine_Loss * tf.reduce_sum(
        weights_foot * tf.reduce_sum(tf.square(j2ds_foot - j2ds_foot_est), 1))

    # TODO try L1, L2 or other penalty function
    objs['Prior_Loss'] = Util.Prior_Loss_refine * tf.reduce_sum(tf.square(param_poses - poses[:frame_num, 3:72]))
    objs['Prior_Shape'] = 5.0 * tf.reduce_sum(tf.square(param_shapes))

    w1 = np.array([1.04 * 2.0, 1.04 * 2.0, 5.4 * 2.0, 5.4 * 2.0])
    w1 = tf.constant(w1, dtype=tf.float32)
    # objs["angle_elbow_knee"] = 0.008 * tf.reduce_sum(w1 * [
    #     tf.exp(param_poses[:, 52]), tf.exp(-param_poses[:, 55]),
    #     tf.exp(-param_poses[:, 9]), tf.exp(-param_poses[:, 12])])
    objs["angle_elbow_knee"] = 0.005 * tf.reduce_sum(w1[0] *
                                                    tf.exp(param_poses[:, 52]) + w1[1] *
                                                    tf.exp(-param_poses[:, 55]) + w1[2] *
                                                    tf.exp(-param_poses[:, 9]) + w1[3] *
                                                    tf.exp(-param_poses[:, 12]))

    param_pose_full = tf.concat([param_rots, param_poses], axis=1)  ## N * 72
    objs['hmr_constraint'] = Util.hmr_constraint_refine * tf.reduce_sum(
        tf.square(tf.squeeze(param_pose_full) - hmr_thetas))

    w_temporal = [0.5, 0.5, 1.0, 1.5, 2.5, 2.5, 1.5, 1.0, 1.0, 1.5, 2.5, 2.5, 1.5, 1.0, 7.0, 7.0]
    for i in range(frame_num - 1):
        j3d_old = j3ds[i, :, :]
        j3d = j3ds[i + 1, :, :]
        j3d_old_tmp = tf.reshape(j3d_old, [-1, 3])  ## (N*16) * 3
        j2d_old = cam.project(tf.squeeze(j3d_old_tmp))  ## (N*16) * 2
        j3d_tmp = tf.reshape(j3d, [-1, 3])  ## (N*16) * 3
        j2d = cam.project(tf.squeeze(j3d_tmp))  ## (N*16) * 2
        param_pose_old = param_poses[i, :]
        param_pose = param_poses[i + 1, :]
        if i == 0:
            objs['temporal3d'] = Util.temporal3d_refine * tf.reduce_sum(
                w_temporal * tf.reduce_sum(tf.square(j3d - j3d_old), 1))
            objs['temporal2d'] = Util.temporal2d_refine * tf.reduce_sum(
                w_temporal * tf.reduce_sum(tf.square(j2d - j2d_old), 1))
            objs['temporal_pose'] = Util.temporal_pose_refine * tf.reduce_sum(
                tf.square(param_pose_old - param_pose))
        else:
            objs['temporal3d'] = objs['temporal3d'] + Util.temporal3d_refine * tf.reduce_sum(
                w_temporal * tf.reduce_sum(tf.square(j3d - j3d_old), 1))
            objs['temporal2d'] = objs['temporal2d'] + Util.temporal2d_refine * tf.reduce_sum(
                w_temporal * tf.reduce_sum(tf.square(j2d - j2d_old), 1))
            objs['temporal_pose'] = objs['temporal_pose'] + Util.temporal_pose_refine * tf.reduce_sum(
                tf.square(param_pose_old - param_pose))

        # TODO add optical flow constraint
        # body_idx = np.array(body_parsing_idx[0]).squeeze()
        # body_idx = body_idx.reshape([-1, 1]).astype(np.int64)
        # verts_est_body = tf.gather_nd(verts_est, body_idx)
        # optical_ratio = 0.0
        # objs['dense_optflow'] = util.params["LR_parameters"]["dense_optflow"] * tf.reduce_sum(tf.square(
        #     verts_est_body - verts_body_old))

    # optimization process
    loss = tf.reduce_sum(objs.values())
    duration = time.time() - start_time
    print("pre-processing time is %f" % duration)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        optimizer = scipy_pt(loss=loss,
                             var_list=[param_shapes, param_rots, param_trans, param_poses, cam.cx, cam.cy],
                             options={'eps': 1e-6, 'ftol': 1e-6, 'maxiter': 10000, 'disp': False})
        print(">>>>>>>>>>>>>start to optimize<<<<<<<<<<<")
        start_time = time.time()
        optimizer.minimize(sess)
        duration = time.time() - start_time
        print("minimize is %f" % duration)
        start_time = time.time()
        poses_final, betas_final, trans_final, cam_cx, cam_cy, v_final, verts_est_final, j3ds_final, _objs = sess.run(
            [tf.concat([param_rots, param_poses], axis=1),
             param_shapes, param_trans, cam.cx, cam.cy, v, verts_est, j3ds, objs])
        v_final = v_final.reshape([frame_num, 6890, 3])
        duration = time.time() - start_time
        print("run time is %f" % duration)
        start_time = time.time()
        cam_for_save = np.array([LR_cameras[0][0], cam_cx, cam_cy, np.zeros(3)])
        ### no sense
        LR_cameras = []
        for i in range(frame_num):
            LR_cameras.append(cam_for_save)
        #############
        camera = render.camera(cam_for_save[0], cam_for_save[1], cam_for_save[2], cam_for_save[3], Util.img_widthheight)

        output_path = Util.hmr_path + Util.refine_output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(output_path + "output_mask"):
            os.makedirs(output_path + "output_mask")
        videowriter = []
        for ind in range(frame_num):
            print(">>>>>>>>>>>>>>>>>>>>>>%d index frame<<<<<<<<<<<<<<<<<<<<<<" % ind)
            if Util.mode == "full":
                smpl = smpl_np.SMPLModel('./smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
                template = np.load(Util.texture_path + "template.npy")
                smpl.set_template(template)
                v = smpl.get_verts(poses_final[ind, :], betas_final[ind, :], trans_final[ind, :])
                texture_vt = np.load(Util.texture_path + "vt.npy")
                texture_img = cv2.imread(Util.texture_path + "../../output_nonrigid/texture.png")
                img_result_texture = camera.render_texture(v, texture_img, texture_vt)
                cv2.imwrite(output_path + "/hmr_optimization_texture_%04d.png" % ind, img_result_texture)
                img_bg = cv2.resize(imgs[ind], (Util.img_width, Util.img_height))
                img_result_texture_bg = camera.render_texture_imgbg(img_result_texture, img_bg)
                cv2.imwrite(output_path + "/texture_bg_%04d.png" % ind,
                            img_result_texture_bg)
                if Util.video is True:
                    if ind == 0:
                        fps = 15
                        size = (imgs[0].shape[1], imgs[0].shape[0])
                        video_path = output_path + "/texture.mp4"
                        videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
                    videowriter.write(img_result_texture)

                img_result_naked = camera.render_naked(v, imgs[ind])
                img_result_naked = img_result_naked[:, :, :3]
                cv2.imwrite(output_path + "/hmr_optimization_%04d.png" % ind, img_result_naked)
                bg = np.ones_like(imgs[ind]).astype(np.uint8) * 255
                img_result_naked1 = camera.render_naked(v, bg)
                cv2.imwrite(output_path + "/hmr_optimization_naked_%04d.png" % ind, img_result_naked1)
                img_result_naked_rotation = camera.render_naked_rotation(v, 90, imgs[ind])
                cv2.imwrite(output_path + "/hmr_optimization_rotation_%04d.png" % ind,
                            img_result_naked_rotation)
                res = {'pose': poses_final[ind, :], 'betas': betas_final[ind, :], 'trans': trans_final[ind, :],
                       'cam_HR': cam_for_save, 'j3ds': j3ds_final[ind, :]}
                with open(output_path + "/hmr_optimization_pose_%04d.pkl" % ind, 'wb') as fout:
                    pkl.dump(res, fout)

                # for z in range(len(verts_est_final)):
                #     if int(verts_est_final[z][0]) > masks[ind].shape[0] - 1:
                #         verts_est_final[z][0] = masks[ind].shape[0] - 1
                #     if int(verts_est_final[z][1]) > masks[ind].shape[1] - 1:
                #         verts_est_final[z][1] = masks[ind].shape[1] - 1
                #     (masks[ind])[int(verts_est_final[z][0]), int(verts_est_final[z][1])] = 127
                # cv2.imwrite(Util.hmr_path + "output_mask/%04d.png" % ind, masks[ind])

            if Util.mode == "pose":
                img_result_naked = camera.render_naked(v_final[ind, :, :], imgs[ind])
                img_result_naked = img_result_naked[:, :, :3]
                cv2.imwrite(output_path + "/hmr_optimization_%04d.png" % ind, img_result_naked)
                if Util.video is True:
                    if ind == 0:
                        fps = 15
                        size = (imgs[0].shape[1], imgs[0].shape[0])
                        video_path = output_path + "/texture.mp4"
                        videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
                    videowriter.write(img_result_naked)
                bg = np.ones_like(imgs[ind]).astype(np.uint8) * 255
                img_result_naked1 = camera.render_naked(v_final[ind, :, :], bg)
                cv2.imwrite(output_path + "/hmr_optimization_naked_%04d.png" % ind, img_result_naked1)
                img_result_naked_rotation = camera.render_naked_rotation(v_final[ind, :, :], 90, imgs[ind])
                cv2.imwrite(output_path + "/hmr_optimization_rotation_%04d.png" % ind,
                            img_result_naked_rotation)
                res = {'pose': poses_final[ind, :], 'betas': betas_final[ind, :], 'trans': trans_final[ind, :],
                       'cam': cam_for_save, 'j3ds': j3ds_final[ind, :]}
                with open(output_path + "/hmr_optimization_pose_%04d.pkl" % ind, 'wb') as fout:
                    pkl.dump(res, fout)

                # for z in range(len(verts_est_final)):
                #     if int(verts_est_final[z][0]) > masks[ind].shape[0] - 1:
                #         verts_est_final[z][0] = masks[ind].shape[0] - 1
                #     if int(verts_est_final[z][1]) > masks[ind].shape[1] - 1:
                #         verts_est_final[z][1] = masks[ind].shape[1] - 1
                #     (masks[ind])[int(verts_est_final[z][0]), int(verts_est_final[z][1])] = 127
                # cv2.imwrite(Util.hmr_path + "output_mask/%04d.png" % ind, masks[ind])
        for name in _objs:
            print("the %s loss is %f" % (name, _objs[name]))
    duration = time.time() - start_time
    print("post-processing time is %f" % duration)
    duration = time.time() - start_time_total
    print("total time is %f" % duration)


def Bundle_Adjustment_motion_refine_optimization(params, hr_points, lr_points):
    Util = Utility()
    Util.read_utility_parameters(params)
    hmr_dict, data_dict = Util.load_hmr_data()
    LR_cams = Util.load_camera_pkl()
    LR_cams = LR_cams[0]
    texture_img = cv2.imread(Util.texture_path + "../../output_nonrigid/texture.png")
    texture_vt = np.load(Util.texture_path + "vt.npy")

    LR_path = Util.hmr_path + Util.refine_reference_path
    LR_pkl_files = os.listdir(LR_path)
    LR_pkl_files = sorted([filename for filename in LR_pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int((d.split('_')[3]).split('.')[0]))
    LR_length = len(LR_pkl_files)
    LR_array = np.zeros((LR_length, 24 * 3))
    LR_betas = np.zeros([LR_length, 10])
    LR_trans = np.zeros([LR_length, 3])
    HR_path = Util.HR_pose_path
    HR_pkl_files = os.listdir(HR_path)
    HR_pkl_files = sorted([filename for filename in HR_pkl_files if filename.endswith(".pkl")],
                          key=lambda d: int((d.split('_')[3]).split('.')[0]))
    HR_length = len(HR_pkl_files)
    HR_array = np.zeros((HR_length, 24 * 3))

    for ind, LR_pkl_file in enumerate(LR_pkl_files):
        LR_pkl_path = os.path.join(LR_path, LR_pkl_file)
        with open(LR_pkl_path) as f:
            param = pickle.load(f)
        pose = param['pose'].squeeze()
        beta = param['betas'].squeeze()
        tran = param['trans'].squeeze()
        for i in range(24 * 3):
            LR_array[ind, i] = pose[i]
        for i in range(10):
            LR_betas[ind, i] = beta[i]
        for i in range(3):
            LR_trans[ind, i] = tran[i]
    for ind, HR_pkl_file in enumerate(HR_pkl_files):
        HR_pkl_path = os.path.join(HR_path, HR_pkl_file)
        with open(HR_pkl_path) as f:
            param = pickle.load(f)
        pose = param['pose'].squeeze()
        for i in range(24 * 3):
            HR_array[ind, i] = pose[i]
    output = algorithms.periodicDecomp(LR_array, HR_array, lr_points, hr_points)
    refine_optimization(output, LR_betas, LR_trans, data_dict, hmr_dict, LR_cams, texture_img, texture_vt, Util)

def main():
    with open("./people2.json", "r") as f:
        params = json.loads(f.read())
    # str_hr3 = "[11,27]"
    # str_lr3 = "[]"
    # str_hr3.append("[11,27]")
    # str_lr3.append("[0,15,30,46,62,76,90,104,119,]")
    # str_hr3.append("[11,27]")
    # str_lr3.append("[0,14,28,42,59,73,87]")
    # str_hr3.append("[14,30]")
    # str_lr3.append("[0,14,31,47,63,78]")
    # str_hr3.append("[10,26]")
    # str_lr3.append("[0,16,33,49,66,82]")
    # str_hr3.append("[2,18]")
    # str_lr3.append("[0,16,32,48,64,80]")
    # str_hr3.append("[11,27]")
    # str_lr3.append("[0,17,33,49,65,81]")
    # str_hr3.append("[3,19]")
    # str_lr3.append("[0,16,32,48,64,80,96]")
    #
    # str_hr = []
    # str_lr = []
    # str_hr.append("[7,23]")
    # str_lr.append("[0,15,30,45,60,75,90]")
    # str_hr.append("[6,22]")
    # str_lr.append("[0,14,28,42,56,70,84]")
    # str_hr.append("[6,22]")
    # str_lr.append("[0,15,30,45,60,75,90]")
    # str_hr.append("[0,16]")
    # str_lr.append("[0,17,34,51,68,85]")
    # str_hr.append("[6,22]")
    # str_lr.append("[0,17,33,49,65,81]")
    # str_hr.append("[14,30]")
    # str_lr.append("[0,17,34,51,68,85]")
    # str_hr.append("[3,19]")
    # str_lr.append("[0,17,34,51,68,85]")
    # str_hr.append("[10,26]")
    # str_lr.append("[0,17,34,51,68,85]")

    str_hr1 = "[11,28]"
    #str_lr1 = "[0,15,30,46,62,76,90,104,119,134,149,164,179,194,209,224,239,254,269,284,299,314,329,344,359,374,389,404,419,434,449,464,479,494,509,524,539,554,569,584,599,614]"
    str_lr1 = "[0,16,32,48,64,80,96,112,128,144,160,176,192]"
    hr_points = np.array((str_hr1).strip().strip('[]').split(",")).astype(int)
    lr_points = np.array((str_lr1).strip().strip('[]').split(",")).astype(int)
    Bundle_Adjustment_motion_refine_optimization(params, hr_points, lr_points)
if __name__ == '__main__':
    main()