import Utility as util_func
from Utility import Utility
from smpl_batch import SMPL
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
from camera import Perspective_Camera
import numpy as np
import algorithms
from opendr_render import render
import time
import os
import cv2
import pickle as pkl
import smpl_np
import json
import sys
def Bundle_Adjustment_optimization(params):
    '''
    Using Bundle Adjustment optimize SMPL parameters with tensorflow-gpu
    :param hmr_dict:
    :param data_dict:
    :return: results
    '''
    start_time = time.time()
    start_time_total = time.time()
    Util = Utility()
    Util.read_utility_parameters(params)
    smpl_model = SMPL(Util.SMPL_COCO_PATH, Util.SMPL_NORMAL_PATH)
    j3dss, success = Util.load_pose_pkl()
    hmr_dict, data_dict = Util.load_hmr_data()

    hmr_thetas = hmr_dict["hmr_thetas"]
    hmr_betas = hmr_dict["hmr_betas"]
    hmr_trans = hmr_dict["hmr_trans"]
    hmr_cams = hmr_dict["hmr_cams"]
    hmr_joint3ds = hmr_dict["hmr_joint3ds"]

    j2ds = data_dict["j2ds"]
    confs = data_dict["confs"]
    j2ds_face = data_dict["j2ds_face"]
    confs_face = data_dict["confs_face"]
    j2ds_head = data_dict["j2ds_head"]
    confs_head = data_dict["confs_head"]
    j2ds_foot = data_dict["j2ds_foot"]
    confs_foot = data_dict["confs_foot"]
    imgs = data_dict["imgs"]
    masks = data_dict["masks"]

    Util.img_width = imgs[0].shape[1]
    Util.img_height = imgs[0].shape[0]
    Util.img_widthheight = int("1" + "%04d" % Util.img_width + "%04d" % Util.img_height)

    frame_num = len(j2ds)

    for ind in range(frame_num):
        hmr_theta = hmr_thetas[ind, :].squeeze()
        # hmr_shape = hmr_betas[ind, :].squeeze()
        # hmr_tran = hmr_trans[ind, :].squeeze()
        # hmr_cam = hmr_cams[0, :].squeeze()
        hmr_joint3d = hmr_joint3ds[ind, :, :]
        ######### Arm Correction #########
        # if Util.pedestrian_constraint == True and success == True:
        #     prej3d = j3dss[ind]
        #     if abs(prej3d[2, 2] - prej3d[7, 2]) > 0.1:
        #         print("leg_error>0.1")
        #         if prej3d[2, 2] < prej3d[7, 2]:
        #             hmr_thetas[ind][51] = 0.8
        #             hmr_theta[52] = 1e-8
        #             hmr_theta[53] = 1.0
        #             hmr_theta[58] = 1e-8
        #             forward_arm = "left"
        #         else:
        #             hmr_theta[48] = 0.8
        #             hmr_theta[49] = 1e-8
        #             hmr_theta[50] = -1.0
        #             hmr_theta[55] = 1e-8
        #             forward_arm = "right"

        if Util.pedestrian_constraint == True:
            if abs(hmr_joint3ds[ind, 0, 2] - hmr_joint3ds[ind, 5, 2]) > 0.1:
                print("leg_error>0.1")
                if hmr_joint3ds[ind, 0, 2] < hmr_joint3ds[ind, 5, 2]:
                    hmr_thetas[ind, 51] = 0.8
                    hmr_thetas[ind, 52] = 1e-8
                    hmr_thetas[ind, 53] = 1.0
                    hmr_thetas[ind, 58] = 1e-8
                    forward_arm = "left"
                else:
                    hmr_thetas[ind, 48] = 0.8
                    hmr_thetas[ind, 49] = 1e-8
                    hmr_thetas[ind, 50] = -1.0
                    hmr_thetas[ind, 55] = 1e-8
                    forward_arm = "right"

    initial_param, pose_mean, pose_covariance = Util.load_initial_param()
    pose_mean = tf.constant(pose_mean, dtype=tf.float32)
    pose_covariance = tf.constant(pose_covariance, dtype=tf.float32)
    ###batch input
    batch_size = frame_num / Util.batch_num
    for ind_batch in range(Util.batch_num):
        ind_start = ind_batch * batch_size
        ind_end = (ind_batch+1) * batch_size

        print(ind_start, ind_end)

        param_shapes = tf.Variable(hmr_betas[ind_start:ind_end,:].reshape([-1, 10]), dtype=tf.float32)
        param_rots = tf.Variable(hmr_thetas[ind_start:ind_end, :3].reshape([-1, 3]), dtype=tf.float32)
        param_poses = tf.Variable(hmr_thetas[ind_start:ind_end, 3:72].reshape([-1, 69]), dtype=tf.float32)
        param_trans = tf.Variable(hmr_trans[ind_start:ind_end,:].reshape([-1, 3]), dtype=tf.float32)
        initial_param_tf = tf.concat([param_shapes, param_rots, param_poses, param_trans], axis=1)  ## N * (72+10+3)

        hmr_cam = hmr_cams[0, :].squeeze()
        cam = Perspective_Camera(hmr_cam[0], hmr_cam[0], hmr_cam[1],
                                hmr_cam[2], np.zeros(3), np.zeros(3))
        j3ds, v, j3dsplus = smpl_model.get_3d_joints(initial_param_tf, Util.SMPL_JOINT_IDS)

        #### divide into different body parts
        j3ds_body = j3ds[:, 2:, :]
        j3ds_head = j3ds[:, 14:16, :]
        j3ds_foot = j3ds[:, :2, :]
        j3ds_face = j3dsplus[:, 14:19, :]
        j3ds_body = tf.reshape(j3ds_body, [-1, 3]) ## (N*12) * 3
        j3ds_head = tf.reshape(j3ds_head, [-1, 3])  ## (N*2) * 3
        j3ds_foot = tf.reshape(j3ds_foot, [-1, 3])  ## (N*2) * 3
        j3ds_face = tf.reshape(j3ds_face, [-1, 3]) ## (N*5) * 3
        j2ds_body_est = cam.project(tf.squeeze(j3ds_body))  ## (N*14) * 2
        j2ds_head_est = cam.project(tf.squeeze(j3ds_head))  ## (N*2) * 2
        j2ds_foot_est = cam.project(tf.squeeze(j3ds_foot))  ## (N*2) * 2
        j2ds_face_est = cam.project(tf.squeeze(j3ds_face))  ## (N*5) * 2

        v = tf.reshape(v, [-1, 3]) ## (N*6890) * 3
        verts_est_mask = cam.project(tf.squeeze(v)) ## (N*6890) * 2
        verts_est = cam.project(tf.squeeze(v)) ## (N*6890) * 2

        # TODO convert the loss function into batch input
        objs = {}

        j2ds_batch = j2ds[ind_start:ind_end, :, :].reshape([-1, 2])  ## (N*14) * 2
        confs_batch = confs[ind_start:ind_end, :].reshape(-1)  ## N*14
        base_weights = np.array(
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        base_weights = np.tile(base_weights, batch_size) ## N*14
        weights = confs_batch * base_weights ## N*14
        weights = tf.constant(weights, dtype=tf.float32) ## N*14
        objs['J2D_Loss'] = Util.J2D_Loss * tf.reduce_sum(weights * tf.reduce_sum(tf.square(j2ds_body_est - j2ds_batch), 1))

        j2ds_face_batch = j2ds_face[ind_start:ind_end, :, :].reshape([-1, 2]) ## (N*5) * 2
        confs_face_batch = confs_face[ind_start:ind_end, :].reshape(-1)  ## N*5
        base_weights_face = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0])
        base_weights_face = np.tile(base_weights_face, batch_size)  ## N*5
        weights_face = confs_face_batch * base_weights_face
        weights_face = tf.constant(weights_face, dtype=tf.float32)
        objs['J2D_face_Loss'] = Util.J2D_face_Loss * tf.reduce_sum(
            weights_face * tf.reduce_sum(tf.square(j2ds_face_est - j2ds_face_batch), 1))

        j2ds_head_batch = j2ds_head[ind_start:ind_end, :, :].reshape([-1, 2])  ## (N*2) * 2
        confs_head_batch = confs_head[ind_start:ind_end, :].reshape(-1)  ## N*2
        base_weights_head = np.array(
            [1.0, 1.0])
        base_weights_head = np.tile(base_weights_head, batch_size)  ## N*2
        weights_head = confs_head_batch * base_weights_head
        weights_head = tf.constant(weights_head, dtype=tf.float32)
        objs['J2D_head_Loss'] = Util.J2D_head_Loss * tf.reduce_sum(
            weights_head * tf.reduce_sum(tf.square(j2ds_head_batch - j2ds_head_est), 1))

        j2ds_foot_batch = j2ds_foot[ind_start:ind_end, :, :].reshape([-1, 2])  ## (N*2) * 2
        confs_foot_batch = confs_foot[ind_start:ind_end, :].reshape(-1)  ## N*2
        base_weights_foot = np.array(
            [1.0, 1.0])
        base_weights_foot = np.tile(base_weights_foot, batch_size)  ## N*2
        weights_foot = confs_foot_batch * base_weights_foot ## N*2
        weights_foot = tf.constant(weights_foot, dtype=tf.float32)
        objs['J2D_foot_Loss'] = Util.J2D_foot_Loss * tf.reduce_sum(
            weights_foot * tf.reduce_sum(tf.square(j2ds_foot_batch - j2ds_foot_est), 1))

        for i in range(batch_size):
            pose_diff = tf.reshape(param_poses[i, :] - pose_mean, [1, -1])
            if i == 0:
                objs['Prior_Loss'] = 1.0 * tf.squeeze(
                    tf.matmul(tf.matmul(pose_diff, pose_covariance), tf.transpose(pose_diff)))
            else:
                objs['Prior_Loss'] = objs['Prior_Loss'] + 1.0 * tf.squeeze(
                    tf.matmul(tf.matmul(pose_diff, pose_covariance), tf.transpose(pose_diff)))
        objs['Prior_Shape'] = 5.0 * tf.reduce_sum(tf.square(param_shapes))

        w1 = np.array([1.04 * 2.0, 1.04 * 2.0, 5.4 * 2.0, 5.4 * 2.0])
        w1 = tf.constant(w1, dtype=tf.float32)
        # objs["angle_elbow_knee"] = 0.008 * tf.reduce_sum(w1 * [
        #     tf.exp(param_poses[:, 52]), tf.exp(-param_poses[:, 55]),
        #     tf.exp(-param_poses[:, 9]), tf.exp(-param_poses[:, 12])])
        objs["angle_elbow_knee"] = 0.08 * tf.reduce_sum(w1[0] *
            tf.exp(param_poses[:, 52]) + w1[1] *
            tf.exp(-param_poses[:, 55]) + w1[2] *
            tf.exp(-param_poses[:, 9]) + w1[3] *
            tf.exp(-param_poses[:, 12]))

        # TODO add a function that deal with masks with batch
        tmp_batch = []
        for i in range(ind_start, ind_end):
            verts2dsilhouette = algorithms.verts_to_silhouette_tf(verts_est_mask, masks[i].shape[1], masks[i].shape[0])
            tmp_batch.append(verts2dsilhouette)
        verts2dsilhouette_batch = tf.convert_to_tensor(tmp_batch)
        mask = np.array(masks[ind_start:ind_end])
        masks_tf = tf.cast(tf.convert_to_tensor(mask), dtype=tf.float32)
        objs['mask'] = Util.mask * tf.reduce_sum(
            verts2dsilhouette_batch / 255.0 * (255.0 - masks_tf) / 255.0
            + (255.0 - verts2dsilhouette_batch) / 255.0 * masks_tf / 255.0)


        # TODO try L1, L2 or other penalty function
        param_pose_full = tf.concat([param_rots, param_poses], axis=1) ## N * 72
        objs['hmr_constraint'] = Util.hmr_constraint * tf.reduce_sum(
            tf.square(tf.squeeze(param_pose_full) - hmr_thetas[ind_start:ind_end,:]))

        objs['hmr_hands_constraint'] = Util.hmr_hands_constraint * tf.reduce_sum(
            tf.square(tf.squeeze(param_pose_full)[:, 21] - hmr_thetas[ind_start:ind_end, 21])
            + tf.square(tf.squeeze(param_pose_full)[:, 23] - hmr_thetas[ind_start:ind_end, 23])
            + tf.square(tf.squeeze(param_pose_full)[:, 20] - hmr_thetas[ind_start:ind_end, 20])
            + tf.square(tf.squeeze(param_pose_full)[:, 22] - hmr_thetas[ind_start:ind_end, 22]))

        # w_temporal = [0.5, 0.5, 1.0, 1.5, 2.5, 2.5, 1.5, 1.0, 1.0, 1.5, 2.5, 2.5, 1.5, 1.0, 7.0, 7.0]
        # for i in range(frame_num-1):
        #     j3d_old = j3ds[i, :, :]
        #     j3d = j3ds[i + 1, :, :]
        #     j3d_old_tmp = tf.reshape(j3d_old, [-1, 3])  ## (N*16) * 3
        #     j2d_old = cam.project(tf.squeeze(j3d_old_tmp))  ## (N*16) * 2
        #     j3d_tmp = tf.reshape(j3d, [-1, 3])  ## (N*16) * 3
        #     j2d = cam.project(tf.squeeze(j3d_tmp))  ## (N*16) * 2
        #     param_pose_old = param_poses[i, :]
        #     param_pose = param_poses[i+1, :]
        #     if i == 0:
        #         objs['temporal3d'] = Util.temporal3d * tf.reduce_sum(
        #             w_temporal * tf.reduce_sum(tf.square(j3d - j3d_old), 1))
        #         objs['temporal2d'] = Util.temporal2d * tf.reduce_sum(
        #             w_temporal * tf.reduce_sum(tf.square(j2d - j2d_old), 1))
        #         objs['temporal_pose'] = Util.temporal_pose * tf.reduce_sum(
        #             tf.square(param_pose_old - param_pose))
        #     else:
        #         objs['temporal3d'] = objs['temporal3d'] + Util.temporal3d * tf.reduce_sum(
        #             w_temporal * tf.reduce_sum(tf.square(j3d - j3d_old), 1))
        #         objs['temporal2d'] = objs['temporal2d'] + Util.temporal2d * tf.reduce_sum(
        #             w_temporal * tf.reduce_sum(tf.square(j2d - j2d_old), 1))
        #         objs['temporal_pose'] = objs['temporal_pose'] + Util.temporal_pose * tf.reduce_sum(
        #             tf.square(param_pose_old - param_pose))

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
                                 options={'eps': 1e-6, 'ftol': 1e-6, 'maxiter': 10000, 'disp': True})
            print(">>>>>>>>>>>>>start to optimize<<<<<<<<<<<")
            start_time = time.time()
            optimizer.minimize(sess)
            duration = time.time() - start_time
            print("minimize is %f" % duration)
            start_time = time.time()
            poses_final, betas_final, trans_final, cam_cx, cam_cy, v_final, verts_est_final, j3ds_final, _objs = sess.run([tf.concat([param_rots, param_poses], axis=1),
                                        param_shapes, param_trans, cam.cx, cam.cy, v, verts_est, j3ds, objs])
            v_final = v_final.reshape([batch_size, 6890, 3])
            duration = time.time() - start_time
            print("run time is %f" % duration)
            start_time = time.time()
            cam_for_save = np.array([hmr_cam[0], cam_cx, cam_cy, np.zeros(3)])
            ### no sense
            LR_cameras = []
            for i in range(ind_start, ind_end):
                LR_cameras.append(cam_for_save)
            #############
            camera = render.camera(cam_for_save[0], cam_for_save[1], cam_for_save[2], cam_for_save[3], Util.img_widthheight)

            output_path = Util.hmr_path + Util.output_path
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if not os.path.exists(Util.hmr_path + "output_mask"):
                os.makedirs(Util.hmr_path + "output_mask")
            videowriter = []
            for i, ind in enumerate(range(ind_start, ind_end)):
                print(">>>>>>>>>>>>>>>>>>>>>>%d index frame<<<<<<<<<<<<<<<<<<<<<<" % ind)
                if Util.mode == "full":
                    # smpl = smpl_np.SMPLModel('./smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
                    # template = np.load(Util.texture_path + "template.npy")
                    # smpl.set_template(template)
                    # v = smpl.get_verts(poses_final[i, :], betas_final[i, :], trans_final[i, :])
                    # texture_vt = np.load(Util.texture_path + "vt.npy")
                    # texture_img = cv2.imread(Util.texture_path + "../../output_nonrigid/texture.png")
                    # img_result_texture = camera.render_texture(v, texture_img, texture_vt)
                    # cv2.imwrite(output_path + "/hmr_optimization_texture_%04d.png" % ind, img_result_texture)
                    # img_bg = cv2.resize(imgs[ind], (Util.img_width, Util.img_height))
                    # img_result_texture_bg = camera.render_texture_imgbg(img_result_texture, img_bg)
                    # cv2.imwrite(output_path + "/texture_bg_%04d.png" % ind,
                    #             img_result_texture_bg)
                    #
                    # img_result_naked = camera.render_naked(v, imgs[ind])
                    # img_result_naked = img_result_naked[:, :, :3]
                    # cv2.imwrite(output_path + "/hmr_optimization_%04d.png" % ind, img_result_naked)
                    # bg = np.ones_like(imgs[ind]).astype(np.uint8) * 255
                    # img_result_naked1 = camera.render_naked(v, bg)
                    # cv2.imwrite(output_path + "/hmr_optimization_naked_%04d.png" % ind, img_result_naked1)
                    # img_result_naked_rotation = camera.render_naked_rotation(v, 90, imgs[ind])
                    # cv2.imwrite(output_path + "/hmr_optimization_rotation_%04d.png" % ind,
                    #             img_result_naked_rotation)
                    res = {'pose': poses_final[i, :], 'betas': betas_final[i, :], 'trans': trans_final[i, :],
                           'cam': cam_for_save, 'j3ds': j3ds_final[i, :]}
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
                    # img_result_naked = camera.render_naked(v_final[i, :, :], imgs[ind])
                    # img_result_naked = img_result_naked[:, :, :3]
                    # cv2.imwrite(output_path + "/hmr_optimization_%04d.png" % ind, img_result_naked)
                    # bg = np.ones_like(imgs[ind]).astype(np.uint8) * 255
                    # img_result_naked1 = camera.render_naked(v_final[i, :, :], bg)
                    # cv2.imwrite(output_path + "/hmr_optimization_naked_%04d.png" % ind, img_result_naked1)
                    # img_result_naked_rotation = camera.render_naked_rotation(v_final[i, :, :], 90, imgs[ind])
                    # cv2.imwrite(output_path + "/hmr_optimization_rotation_%04d.png" % ind,
                    #             img_result_naked_rotation)
                    res = {'pose': poses_final[i, :], 'betas': betas_final[i, :], 'trans': trans_final[i, :],
                           'cam': cam_for_save, 'j3ds': j3ds_final[i, :]}
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
    if Util.video is True:
        fps = 15
        size = (imgs[0].shape[1], imgs[0].shape[0])
        video_path = output_path + "/texture.mp4"
        videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
        if Util.mode == "full":
            for i in range(batch_size*Util.batch_num):
                img = cv2.imread(output_path + "/texture_bg_%04d.png" % ind)
                videowriter.write(img)
        if Util.mode == "pose":
            for i in range(batch_size*Util.batch_num):
                img = cv2.imread(output_path + "/hmr_optimization_%04d.png" % ind)
                videowriter.write(img)
    util_func.save_pkl_to_csv(output_path)
    util_func.save_pkl_to_npy(output_path)
    duration = time.time() - start_time
    print("post-processing time is %f" % duration)
    duration = time.time() - start_time_total
    print("total time is %f" % duration)


def main():
    with open(sys.argv[1], "r") as f:
        params = json.loads(f.read())
    print(">>>>>>>>>>>>read success!!!!!<<<<<<<<<<")
    Bundle_Adjustment_optimization(params)
if __name__ == '__main__':
    main()
