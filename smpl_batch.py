#coding=utf-8

import tensorflow as tf
import numpy as np
import Utility
import pickle as pkl
from psbody.meshlite import Mesh
from smpl.smpl_webuser.serialization import load_model
import smpl_np

class SMPL:
	# Read model paramters
	def __init__(self, smpl_model_path, normal_smpl_model_path):
		data = pkl.load(open(smpl_model_path))
		data1 = pkl.load(open(normal_smpl_model_path))
		template = data1['v_template']
		#v_template = smpl_np.remove_template_handfoot(template, data1['weights'])
		self.v_template = tf.constant(template, dtype=tf.float32)
		self.v_template_RT = tf.constant(data1['v_template'], dtype=tf.float32)
		self.f = tf.constant(data1['f'], dtype=tf.float32)
		self.shapedirs = tf.constant(data1['shapedirs'].r, dtype=tf.float32)
		self.posedirs = tf.constant(data1['posedirs'], dtype=tf.float32)
		self.J_regressor = tf.constant(data1['J_regressor'].todense(), dtype=tf.float32)
		self.parent_ids = data1['kintree_table'][0].astype(int)
		self.weights = tf.constant(data1['weights'], dtype=tf.float32)
		self.joint_regressor = tf.constant(data['cocoplus_regressor'].T.todense(), dtype=tf.float32)

	### no useful temporarily
	def set_template(self, template):
		self.v_template = tf.constant(template, dtype=tf.float32)

	# N x 
	def get_3d_joints(self, params, index_list):
		betas = params[:, :10]
		pose = params[:, 10:82]
		trans = params[:, -3:]
		smpl_animated = self.animate(betas, pose, trans)
		A_global = smpl_animated['A_global']
		v_head = tf.squeeze(smpl_animated['v'][:, Utility.HEAD_VID]) + trans
		A_global = tf.convert_to_tensor(A_global)
		A_global = tf.transpose(A_global, [1, 0, 2, 3])
		res =  A_global[:, :, :3, 3] + tf.expand_dims(trans, axis=1) # N x 24 x 3
		tmp = []
		for i in index_list:
			tmp.append(res[:, i]) # N x 3
		res = tf.convert_to_tensor(tmp)
		res = tf.transpose(res, [1, 0, 2])
		res = tf.concat([res, tf.expand_dims(v_head, axis=1)], axis=1)

		verts = smpl_animated['v']
		joint_x = tf.matmul(verts[:, :, 0], self.joint_regressor)
		joint_y = tf.matmul(verts[:, :, 1], self.joint_regressor)
		joint_z = tf.matmul(verts[:, :, 2], self.joint_regressor)
		joints = tf.stack([joint_x, joint_y, joint_z], axis=2)
		#joints = joints + tf.reshape(trans, [-1, 1, 3])

		return res, smpl_animated['v'] + tf.reshape(trans, [-1, 1, 3]), joints + tf.reshape(trans, [-1, 1, 3])

	# Turn pose into global orientation, joints and mesh
	# betas: N x 10
	# pose: N x 72
	# trans: N x 3
	def animate(self, betas, pose, trans):
		total = pose.get_shape().as_list()[0]
		betas = tf.cast(betas, dtype=tf.float32)
		pose = tf.cast(pose, dtype=tf.float32)
		trans = tf.cast(trans, dtype=tf.float32)

		pose_rest = tf.zeros([23, 3], dtype=tf.float32)
		# For current pose
		pose_matrix_cur = Rodrigues(tf.reshape(pose[:, 3:], [-1, 3])) 
		pose_matrix_cur = tf.reshape(pose_matrix_cur, [-1, 23*9])
		# For rest pose
		pose_matrix_rest = Rodrigues(pose_rest) 
		pose_matrix_rest = tf.reshape(pose_matrix_rest, [1, -1])
		# Subtraction
		pose_matrix_cur = pose_matrix_cur - pose_matrix_rest
		pose_matrix = pose_matrix_cur #现在的pose旋转矩阵减去0 pose旋转矩阵
		

		v_shaped_v = tf.expand_dims(self.v_template, axis=0) + \
			tf.einsum('ijk,lk->lij', self.shapedirs, betas)  # 10 x 6890 x 3
		v_shaped = tf.expand_dims(self.v_template_RT, axis=0) + \
				   tf.einsum('ijk,lk->lij', self.shapedirs, betas)  # 10 x 6890 x 3
		#相当于生成一个关于beta的模
		v_posed = v_shaped_v + tf.einsum('ijk,lk->lij', self.posedirs, pose_matrix)

		# sess = tf.Session()
		# sess.run(tf.global_variables_initializer())
		# v_posed = sess.run(v_posed)
		# v_posed = v_posed.squeeze()
		# import matplotlib.pyplot as plt
		# from mpl_toolkits.mplot3d import Axes3D
		# fig = plt.figure(1)
		# #ax = plt.subplot(111)
		# ax = fig.add_subplot(111, projection='3d')
		# ax.scatter(v_posed[:, 0], v_posed[:, 1], v_posed[:, 2], c='b')
		# #ax.scatter(HR_j2d[:, 0], HR_j2d[:, 1], c='r')
		# #plt.savefig("/home/lgh/code/SMPLify_TF/test/temp0/1/LR/test_temp/dot.png")
		# plt.show()

		J = tf.matmul(tf.tile(tf.expand_dims(self.J_regressor, axis=0), [total, 1, 1]), v_shaped)
		#J: 1 * 24 * 3,相当于0 pose，参数beta下的3d joint点的三维坐标
		#J = tf.einsum('ij,kjl->kil', self.J_regressor, v_posed)


		pose_mat = Rodrigues(tf.reshape(pose, [-1, 3]))
		pose_mat = tf.reshape(pose_mat, [-1, 24, 3, 3])

                def with_zeros(xs):
		                #相当于加最后一列
                        '''Add [0,0,0,1] into xs
                        Args: 
                                1), xs: Nx3x4, 

                        Yields: 
                                1), Nx4x4
                        '''
                        return tf.concat([xs, tf.constant(np.tile(np.array([0,0,0,1]).reshape([1, 1,4]), [total, 1, 1]), dtype=tf.float32)], axis=1)    

                A_global = [0]*24 # 24, Nx4x4
                #A_global[0] = with_zeros(tf.concat([Rodrigues(tf.constant(np.zeros([total, 3]), dtype=tf.float32)), tf.expand_dims(J[:,0], axis=2)], axis=2))
                A_global[0] = with_zeros(tf.concat([Rodrigues(pose[:, :3]), tf.expand_dims(J[:,0], axis=2)], axis=2))
                for idx, pid in enumerate(self.parent_ids[1:]):
                        A_global[idx+1] = tf.matmul(A_global[pid],
                                                with_zeros(tf.concat([pose_mat[:, idx+1, :3, :3], tf.expand_dims(J[:, idx+1]-J[:, pid], axis=2)], axis=2)))
						#pose_mat:1 * 24 * 3 * 3  pose_mat[:, idx+1, :3, :3]是一个1 * 1 * 3 * 3的矩阵，J[:, idx+1]是1 * 3

		# sess = tf.Session()
		# sess.run(tf.global_variables_initializer())
		# A_global[0] = sess.run(A_global[0])
		# A_global[0] = A_global[0].squeeze()
                def pack(xs): # 24, totalx4x4
                        #return tf.concat([tf.constant(np.zeros([total, 4,3])), tf.reshape(x, [4,1])], axis=1)          
                        return tf.concat([tf.constant(np.zeros([total, 4,3]), dtype=tf.float32), xs], axis=2)
				#从1 * 4 * 3变成1*4*4,1*4*3是全0，最后一个频道是
                A_tmp = [A_global[i] - pack(tf.matmul(A_global[i], tf.expand_dims(tf.concat([J[:, i], tf.constant(np.zeros([total, 1]), dtype=tf.float32)], axis=1), axis=-1))) for i in range(len(A_global))]
                #A_tmp = tf.stack(A_tmp, axis=2)
                A_tmp = tf.convert_to_tensor(A_tmp)
                A_tmp = tf.transpose(A_tmp, [1, 2, 3, 0])
                A_tmp = tf.reshape(A_tmp, [total, 16, 24])

                T = tf.matmul(A_tmp, tf.tile(tf.reshape(tf.transpose(self.weights), [1, 24, 6890]), [total, 1, 1]))
                T = tf.reshape(T, [total, 4, 4, -1])

                rest_shape_h = tf.concat([tf.transpose(v_posed), tf.constant(np.ones([1, 6890, total]), dtype=tf.float32)], axis=0)
                rest_shape_h = tf.transpose(rest_shape_h, [2, 0, 1])
		### T 1,4,4,6890  rest_shape_h 1,4,6890
        # 	b = T[:, :,0,:]
    	# 	c = tf.expand_dims(rest_shape_h[:, 0,:], axis=1)
		# a = tf.multiply(T[:, :,0,:], tf.expand_dims(rest_shape_h[:, 0,:], axis=1))
		v = tf.transpose(tf.multiply(T[:, :,0,:], tf.expand_dims(rest_shape_h[:, 0,:], axis=1)) \
                	+ tf.multiply(T[:, :,1,:], tf.expand_dims(rest_shape_h[:, 1,:],axis=1)) \
                        + tf.multiply(T[:, :,2,:], tf.expand_dims(rest_shape_h[:, 2,:], axis=1)) \
                        + tf.multiply(T[:, :,3,:], tf.expand_dims(rest_shape_h[:, 3,:], axis=1)))
                v = tf.transpose(v, [2, 0, 1])
                v = v[:, :, :3]

		result = {'A_global': A_global, 'v':v}
		return result


def Rodrigues(rot_vs):
        '''Rodrigues in batch mode
        
        Args:
                1), rot_vs: Nx3 rotation vectors

        Yields:
                1), Nx3x3 rotation matrix
        '''
        total = rot_vs.get_shape().as_list()[0]

        a = tf.sqrt(tf.reduce_sum(tf.multiply(rot_vs, rot_vs), axis=1)) + 1e-8
        c = tf.cos(a)
        s = tf.sin(a)

        t = 1 - c
        x = tf.divide(rot_vs[:, 0], a)
        y = tf.divide(rot_vs[:, 1], a)
        z = tf.divide(rot_vs[:, 2], a)
        x = tf.reshape(x, [-1, 1])
        y = tf.reshape(y, [-1, 1])
        z = tf.reshape(z, [-1, 1])


        tmp_0 = tf.stack([tf.multiply(x, x), tf.multiply(x, y), tf.multiply(x, z), tf.multiply(x, y), tf.multiply(y, y), tf.multiply(y, z), tf.multiply(x, z), tf.multiply(y, z), tf.multiply(z, z)], axis=-1)
        tmp_0 = tf.reshape(tmp_0, [total, 3, 3])

        tmp_1 = tf.stack([np.zeros([total, 1]), -1 * z, y, z, np.zeros([total, 1]), -1 * x, -1 * y, x, np.zeros([total, 1])], axis=-1)
        tmp_1 = tf.reshape(tmp_1, [total, 3, 3])

        rot_mats = tf.multiply(tf.reshape(c, [total, 1, 1]), tf.constant(np.tile(np.eye(3).reshape([1, 3, 3]), [total, 1, 1]), dtype=tf.float32)) \
                        + tf.multiply(tf.reshape(t, [total, 1, 1]), tmp_0) \
                        + tf.multiply(tf.reshape(s, [total, 1, 1]), tmp_1) \

        return rot_mats


if __name__ == '__main__':	
	smpl_batch = SMPL('../Data/Smpl_Model/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
	

	sess = tf.Session()
	'''
	tmp = np.random.rand(5, 85)
	params = tf.Variable(tmp, dtype=tf.float32)
	'''
	param_shape = tf.Variable(np.ones([5, 10]) * 1e-8, dtype=tf.float32)
	param_rot = tf.Variable(np.ones([5, 3]) * 1e-8,  dtype=tf.float32)
	param_pose = tf.Variable(np.ones([5, 69]) * 1e-8, dtype=tf.float32)
	param_trans = tf.Variable(np.ones([5, 3]) * 1e-8, dtype=tf.float32)
	params = tf.concat([param_shape, param_rot, param_pose, param_trans], axis=1)
	res_batch = smpl_batch.get_3d_joints(params, range(0, 10))

	sess.run(tf.global_variables_initializer())
	
	objs = {}
	#objs['J2D_Loss'] = tf.reduce_sum( tf.square(j2ds_est - j2ds) )
	j3d_gt = np.random.rand(11, 3)
	objs['J3D_Loss'] = tf.reduce_sum( tf.square(res_batch[0] - j3d_gt) )
	loss = tf.reduce_mean(objs.values())
        #optimizer.minimize(sess, fetches = [j3ds, params, objs, j2ds_est], loss_callback=lc)

	from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
        optimizer = scipy_pt(loss=loss, var_list=[param_rot, param_shape, param_pose, param_trans], options={'ftol':0.001, 'maxiter':500, 'disp':True}, method='L-BFGS-B')
	
	def lc(loss, res):
		return
		m = Mesh(v=res)
		m2 = Mesh(v=j3d_gt)
		m.show()
		m2.show()
		import time
		time.sleep(3)	

        optimizer.minimize(sess, fetches=[objs['J3D_Loss'], res_batch[0]], loss_callback=lc)
	import ipdb; ipdb.set_trace()

