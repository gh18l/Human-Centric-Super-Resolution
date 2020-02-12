import tensorflow as tf
import numpy as np
import scipy.signal as signal
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
def verts_to_silhouette_tf(verts_est_mask, mask_with, mask_height):
    verts_est_mask = tf.cast(verts_est_mask, dtype=tf.int64)
    verts_est_mask = tf.concat([tf.expand_dims(verts_est_mask[:, 1], 1),
                                tf.expand_dims(verts_est_mask[:, 0], 1)], 1)

    verts_est_shape = verts_est_mask.get_shape().as_list()
    temp_np = np.ones([verts_est_shape[0]]) * 255
    temp_np = tf.convert_to_tensor(temp_np, dtype=tf.float32)
    delta_shape = tf.convert_to_tensor([mask_height, mask_with],
                                       dtype=tf.int64)
    scatter = tf.scatter_nd(verts_est_mask, temp_np, delta_shape)
    compare = np.zeros([mask_height, mask_with])
    compare = tf.convert_to_tensor(compare, dtype=tf.float32)
    scatter = tf.not_equal(scatter, compare)
    scatter = tf.cast(scatter, dtype=tf.float32)
    scatter = scatter * tf.convert_to_tensor([255.0], dtype=tf.float32)
    scatter = tf.expand_dims(scatter, 0)
    scatter = tf.expand_dims(scatter, -1)
    ###########kernel###############
    filter = np.zeros([9, 9, 1])
    filter = tf.convert_to_tensor(filter, dtype=tf.float32)
    strides = [1, 1, 1, 1]
    rates = [1, 1, 1, 1]
    padding = "SAME"
    scatter = tf.nn.dilation2d(scatter, filter, strides, rates, padding)
    verts2dsilhouette = tf.nn.erosion2d(scatter, filter, strides, rates, padding)
    # tf.gather_nd(verts2dsilhouette, verts_est) = 255
    verts2dsilhouette = tf.squeeze(verts2dsilhouette)

    return verts2dsilhouette

# def mean_smoothing(s,r):
#     s2=np.zeros(s.shape)
#     len = s.size
#     for i in range(r):
#         temp1 = 0
#         temp2 = 0
#         for j in range(i):
#             temp1 += s[i]
#             temp2 += s[len - i -1]
#         s2[i] = temp1 / (i+1)
#         s2[len - i - 1] = temp2 / (i+1)
#     for i in range(r, len - r):
#         tempSum = 0
#         for j in range(1, r+1):
#             tempSum += (s[i-j] +s[i+j])
#         s2[i]=(s[i]+tempSum) / (2*r + 1)
#     return s2
#
# def exponential_smoothing(s,alpha,r):
#     s2=np.zeros(s.shape)
#     len = s.size
#     for i in range(r):
#         s2[i] = s[i]
#         s2[len - i - 1] = s[len - i -1]
#     beta = (1-alpha) / (r*2)
#     for i in range(r, len - r):
#         tempSum = 0
#         for j in range(1, r+1):
#             tempSum += (s[i-j] +s[i+j])
#         s2[i]=alpha*s[i]+beta*tempSum
#     return s2
#
# def periodicDecomp(lr, hr, lr_points, hr_points):
#     lr = lr
#     hr = hr
#     lr_num = len(lr_points)-1
#     hr_num = len(hr_points)-1
#     lr_len = hr_len = 9999
#     for i in range(lr_num):
#         if lr_points[i+1] - lr_points[i] < lr_len:
#             lr_len = lr_points[i+1] - lr_points[i]
#     for i in range(hr_num):
#         if hr_points[i+1] - hr_points[i] < hr_len:
#             hr_len = hr_points[i+1] - hr_points[i]
#     if hr_len <= lr_len:
#         lr_len = hr_len
#     else:
#         hr_len = lr_len
#     lr_mean = np.mean(lr[lr_points[0]:lr_points[-1]+1], axis=0)
#     hr_mean = np.mean(hr[hr_points[0]:hr_points[-1]+1], axis=0)
#
#     results = []
#     for k in range(72):
#         ### no change rot
#         # if k < 3:
#         #     results.append(np.array(lr[:, k][lr_points[0]:(lr_points[-1])]))
#         #     continue
#         hr_4 = hr[:,k] #here
#         # hr_pSeg = [6,21,36,51, 67,82]
#         hr_pSeg = hr_points
#         hr_4_s = []
#         hr_segLen = []
#         for p in range(hr_num):
#             hr_4_s.append(hr_4[hr_pSeg[p]: hr_pSeg[p+1]])
#             hr_segLen.append((hr_pSeg[p+1]-hr_pSeg[p])/lr_len)
#         hr_part_mean = []
#         for j in range(1,hr_len+1):
#             tempSum = 0
#             tempLen = 0
#             for i in range(hr_num):
#                 tempSum += np.mean(hr_4_s[i][int(hr_segLen[i]*(j-1)):int(hr_segLen[i]*j+1)])
#             hr_part_mean.append(tempSum / hr_num)
#         hr_factor_mul_4 = np.array(hr_part_mean) / hr_mean[k]
#         hr_factor_add_4 = np.array(hr_part_mean) - hr_mean[k]
#
#         lr_4 = lr[:,k] # here
#         # lr_pSeg = [0,13,31,47,61,75,90]
#         lr_pSeg = lr_points
#         lr_4_s = []
#         lr_segLen = []
#         for i in range(len(lr_pSeg) - 1):
#             lr_4_s.append(lr_4[lr_pSeg[i]:lr_pSeg[i+1]])
#             lr_segLen.append((lr_pSeg[i+1] - lr_pSeg[i])/lr_len)
#         lr_part_mean = []
#         for j in range(1,lr_len+1):
#             tempSum = 0
#             tempLen = 0
#             for i in range(lr_num):
#                 tempSum += np.mean(lr_4_s[i][int(lr_segLen[i]*(j-1)):int(lr_segLen[i]*j+1)])
#             lr_part_mean.append(tempSum/lr_num)
#         lr_factor_mul_4 = np.array(lr_part_mean) / lr_mean[k]
#         lr_factor_add_4 = np.array(lr_part_mean) - lr_mean[k]
#         # print(lr_mean[k])
#
#         mline = np.ones([len(lr), 1]) *  lr_mean[k]
#         lr_4_m = []
#         for i in range(len(lr_pSeg) - 1):
#             lr_4_m.append(mline[lr_pSeg[i]:lr_pSeg[i+1]])
#         for j in range(1,lr_len+1):
#             tempSum = 0
#             tempLen = 0
#             for i in range(lr_num):
#                 # print("--j:--",j,"--i:--",i)
#                 # print(lr_4_s[i][int(segLen[i]*(j-1)):int(segLen[i]*j)])
#                 lr_4_m[i][int(lr_segLen[i] * (j - 1)):int(lr_segLen[i] * j)] += hr_factor_add_4[j-1]
#         result = []
#         for i in range(len(lr_4_m)):
#             for j in lr_4_m[i]:
#                 # print(j[0])
#                 result.append(j[0])
#         results.append(np.array(result))
#     output = np.array(results).T
#     # data = pd.DataFrame(output)
#     # data.to_csv('tianyi_pose_0111.csv',header = False, index = False) # here
#     # data.to_csv(output_file,header = False, index = False) # here
#     return output
#
# def max_magnitude(pose):
#     sp = np.fft.fft(pose)
#     mag = np.abs(sp)
#     mag_short = np.array(mag[1:])
#     idx = np.argwhere(mag_short == np.max(mag_short))
#     return idx[0][0]
#
# '''spectral transfer'''
# def spectral_transfer(hr, lr, input, max_idx):
#     output = np.array(input)
#     len_hr = hr.shape[0]
#     per25 = int(len_hr*0.25)
#     per125 = int(len_hr*0.125)
#     for j in range(3,72):
#         '''hr'''
#         pose_hr = hr[:, j]
#         sp_hr = np.fft.fft(pose_hr)
#         real_hr = sp_hr.real
#         imag_hr = sp_hr.imag
#         magnitude_hr = np.abs(sp_hr)
#         phase_hr = np.angle(sp_hr)
#         '''lr'''
#         pose_lr = lr[:, j]
#         sp_lr = np.fft.fft(pose_lr)
#         real_lr = sp_lr.real
#         imag_lr = sp_lr.imag
#         magnitude_lr = np.abs(sp_lr)
#         phase_lr = np.angle(sp_lr)
#         '''different'''
#         mag_d = magnitude_hr - magnitude_lr
#         phase_d = phase_hr - phase_lr
#         real_d = sp_hr.real - sp_lr.real
#         imag_d = sp_hr.imag - sp_lr.imag
#         '''input'''
#         pose_input = input[:, j]
#         sp_input = np.fft.fft(pose_input)
#         real_input = sp_input.real
#         imag_input = sp_input.imag
#         magnitude_input = np.abs(sp_input)
#         phase_input = np.angle(sp_input)
#         '''transfer'''
#         sp = np.array(sp_input)
#         magnitude = np.array(magnitude_hr)
#         magnitude[max_idx] += mag_d[max_idx]
#         magnitude[len_hr-max_idx] += mag_d[len_hr-max_idx]
#         magnitude[max_idx+1: len_hr - (max_idx+1)] = (magnitude[max_idx+1: len_hr - (max_idx+1)] + mag_d[max_idx+1: len_hr - (max_idx+1)] + magnitude_hr[max_idx+1: len_hr - (max_idx+1)])/2
#         magnitude[per25:len_hr-per25] = (magnitude[per25:len_hr-per25] + mag_d[per25:len_hr-per25] + magnitude_hr[per25:len_hr-per25])/4
#         magnitude[per125:len_hr-per125] = (magnitude[per125:len_hr-per125] + mag_d[per125:len_hr-per125] + magnitude_hr[per125:len_hr-per125])/8
#         phase = np.array(phase_input)
#         sp.real = np.abs(magnitude) / np.sqrt(1 + (np.square(np.tan(phase)))) * np.sign(np.cos(phase))
#         sp.imag = sp.real * np.tan(phase)
#         output[:, j] = np.fft.ifft(sp)
#     return output
#
#
#
# def spectral_refine(pose_hr, pose_lr, match_idx):
#     '''cut parts'''
#     len_hr = pose_hr.shape[0]
#     len_lr = pose_lr.shape[0]
#     fore_num = int((match_idx) / len_hr)
#     back_num = int((len_lr - match_idx) / len_hr)
#     first_end = match_idx - fore_num*len_hr
#     last_begin = match_idx + back_num*len_hr
#
#     '''pre'''
#     input = np.array(pose_lr)
#     pose3 = pose_hr[:, 3]
#     max_idx = max_magnitude(pose3) + 1
#     pose_lr = pose_lr[match_idx: match_idx+len_hr, :]
#
#     'every part spectral_transfer'
#     if first_end != 0:
#         pose_input = input[0: first_end, :]
#         pose_output_first = spectral_transfer(pose_hr[-(first_end):, :], pose_lr[-(first_end):, :], pose_input, max_idx)
#         result = np.array(pose_output_first)
#     for i in range(1, fore_num+1):
#         begin_idx = match_idx - i*len_hr
#         pose_input = input[begin_idx: begin_idx + len_hr, :]
#         pose_output = spectral_transfer(pose_hr, pose_lr, pose_input, max_idx)
#         result = np.vstack((result, pose_output))
#     for i in range(back_num):
#         begin_idx = match_idx + i*len_hr
#         pose_input = input[begin_idx: begin_idx + len_hr, :]
#         pose_output = spectral_transfer(pose_hr, pose_lr, pose_input, max_idx)
#         result = np.vstack((result, pose_output))
#     if last_begin < len_lr:
#         pose_input = input[last_begin:, :]
#         pose_output_last = spectral_transfer(pose_hr[0: len_lr-last_begin, :], pose_lr[0: len_lr-last_begin, :], pose_input, max_idx)
#         result = np.vstack((result, pose_output_last))
#     return result

def max_magnitude(pose):
    sp = np.fft.fft(pose)
    mag = np.abs(sp)
    mag_short = np.array(mag[1:])
    # print(mag)
    idx = np.argwhere(mag == np.max(mag_short[:int(mag_short.size/2)]))
    return idx[0][0]

'''spectral transfer'''
def spectral_transfer(hr, lr, input, max_idx):
    output = np.array(input)
    len_hr = hr.shape[0]
    per25 = int(len_hr*0.25)
    per125 = int(len_hr*0.125)
    for j in range(72):
        '''hr'''
        pose_hr = hr[:, j]
        sp_hr = np.fft.fft(pose_hr)
        real_hr = sp_hr.real
        imag_hr = sp_hr.imag
        magnitude_hr = np.abs(sp_hr)
        phase_hr = np.angle(sp_hr)
        '''lr'''
        pose_lr = lr[:, j]
        sp_lr = np.fft.fft(pose_lr)
        real_lr = sp_lr.real
        imag_lr = sp_lr.imag
        magnitude_lr = np.abs(sp_lr)
        phase_lr = np.angle(sp_lr)
        '''different'''
        mag_d = magnitude_hr - magnitude_lr
        phase_d = phase_hr - phase_lr
        real_d = sp_hr.real - sp_lr.real
        imag_d = sp_hr.imag - sp_lr.imag
        '''input'''
        pose_input = input[:, j]
        sp_input = np.fft.fft(pose_input)
        real_input = sp_input.real
        imag_input = sp_input.imag
        magnitude_input = np.abs(sp_input)
        phase_input = np.angle(sp_input)
        '''transfer'''
        sp = np.array(sp_input)
        magnitude = np.array(magnitude_hr)
        magnitude[max_idx] += mag_d[max_idx]
        magnitude[len_hr-max_idx] += mag_d[len_hr-max_idx]
        # magnitude[max_idx+1: len_hr - (max_idx+1)] = (magnitude[max_idx+1: len_hr - (max_idx+1)] + mag_d[max_idx+1: len_hr - (max_idx+1)] + magnitude_hr[max_idx+1: len_hr - (max_idx+1)])/2
        # magnitude[per25:len_hr-per25] = (magnitude[per25:len_hr-per25] + mag_d[per25:len_hr-per25] + magnitude_hr[per25:len_hr-per25])/4
        # magnitude[per125:len_hr-per125] = (magnitude[per125:len_hr-per125] + mag_d[per125:len_hr-per125] + magnitude_hr[per125:len_hr-per125])/8
        magnitude[max_idx+1: len_hr - (max_idx+1)] = (magnitude[max_idx+1: len_hr - (max_idx+1)] + magnitude_hr[max_idx+1: len_hr - (max_idx+1)])/2
        magnitude[per25:len_hr-per25] = (magnitude[per25:len_hr-per25] + magnitude_hr[per25:len_hr-per25])/4
        magnitude[per125:len_hr-per125] = (magnitude[per125:len_hr-per125] + magnitude_hr[per125:len_hr-per125])/8
        phase = np.array(phase_input)
        sp.real = np.abs(magnitude) / np.sqrt(1 + (np.square(np.tan(phase)))) * np.sign(np.cos(phase))
        sp.imag = sp.real * np.tan(phase)
        output[:, j] = np.fft.ifft(sp)
    return output

'''spectral transfer (a period part)'''
def spectral_transfer_p(hr, lr):
    output = np.array(lr)
    len_hr = hr.shape[0]
    per25 = int(len_hr*0.25)
    per125 = int(len_hr*0.125)
    for j in range(72):
        '''hr'''
        pose_hr = hr[:, j]
        sp_hr = np.fft.fft(pose_hr)
        real_hr = sp_hr.real
        imag_hr = sp_hr.imag
        magnitude_hr = np.abs(sp_hr)
        phase_hr = np.angle(sp_hr)
        '''lr'''
        pose_lr = lr[:, j]
        sp_lr = np.fft.fft(pose_lr)
        real_lr = sp_lr.real
        imag_lr = sp_lr.imag
        magnitude_lr = np.abs(sp_lr)
        phase_lr = np.angle(sp_lr)
        '''different'''
        mag_d = magnitude_hr - magnitude_lr
        phase_d = phase_hr - phase_lr
        real_d = sp_hr.real - sp_lr.real
        imag_d = sp_hr.imag - sp_lr.imag
        '''transfer'''
        sp = np.array(sp_lr)
        magnitude = np.array((magnitude_hr + magnitude_lr) / 2)
        # magnitude[max_idx] += mag_d[max_idx]
        # magnitude[len_hr-max_idx] += mag_d[len_hr-max_idx]
        # magnitude[max_idx+1: len_hr - (max_idx+1)] = (magnitude[max_idx+1: len_hr - (max_idx+1)] + mag_d[max_idx+1: len_hr - (max_idx+1)] + magnitude_hr[max_idx+1: len_hr - (max_idx+1)])/2
        # magnitude[per25:len_hr-per25] = (magnitude[per25:len_hr-per25] + mag_d[per25:len_hr-per25] + magnitude_hr[per25:len_hr-per25])/4
        # magnitude[per125:len_hr-per125] = (magnitude[per125:len_hr-per125] + mag_d[per125:len_hr-per125] + magnitude_hr[per125:len_hr-per125])/8
        # magnitude[max_idx+1: len_hr - (max_idx+1)] = (magnitude[max_idx+1: len_hr - (max_idx+1)] + magnitude_hr[max_idx+1: len_hr - (max_idx+1)])/2
        # magnitude[per25:len_hr-per25] = (magnitude[per25:len_hr-per25] + magnitude_hr[per25:len_hr-per25])/4
        # magnitude[per125:len_hr-per125] = (magnitude[per125:len_hr-per125] + magnitude_hr[per125:len_hr-per125])/8
        phase = np.array(phase_hr)
        sp.real = np.abs(magnitude) / np.sqrt(1 + (np.square(np.tan(phase)))) * np.sign(np.cos(phase))
        sp.imag = sp.real * np.tan(phase)
        output[:, j] = np.fft.ifft(sp)
        # output1 = np.fft.ifft(sp)
    return output

'''resample'''
def resample(pose_cur_hr, len):
    x = np.arange(1, pose_cur_hr.shape[0] + 1, 1)
    output = np.zeros((len, pose_cur_hr.shape[1]))
    for j in range(72):
        f = interp1d(x, pose_cur_hr[:, j], kind='cubic')
        x_new = np.linspace(x.min(), x.max(), len)
        output[:, j] = f(x_new)
    return output

def refineLR(pose_lr, pose_hr, match_idx):
    '''cut parts'''
    len_hr = pose_hr.shape[0]
    len_lr = pose_lr.shape[0]
    fore_num = int((match_idx) / len_hr)
    back_num = int((len_lr - match_idx) / len_hr)
    first_end = match_idx - fore_num*len_hr
    last_begin = match_idx + back_num*len_hr

    '''pre'''
    input = np.array(pose_lr)
    pose3_hr = pose_hr[:, 3] - np.mean(pose_hr[:, 3])
    pose6_hr = pose_hr[:, 6] - np.mean(pose_hr[:, 6])
    pose_diff_hr = pose_hr[:, 3] - pose_hr[:, 6]
    pose_diff_hr = pose_diff_hr - np.mean(pose_diff_hr)
    pose3_lr = pose_lr[:, 3] - np.mean(pose_lr[:, 3])
    pose_diff_lr = pose_lr[:, 3] - pose_lr[:, 6]
    pose_diff_lr = pose_diff_lr - np.mean(pose_diff_lr)
    max_idx_hr = max_magnitude(pose_hr[:, 3])
    max_idx_lr = max_magnitude(pose_lr[:, 3])
    # pose_lr = pose_lr[match_idx: match_idx+len_hr, :]

    '''find_period'''
    pose3_smooth = np.fft.fft(pose_diff_hr)
    pose3_mag = np.abs(pose3_smooth)
    pose3_phase = np.angle(pose3_smooth)
    pose3_mag[max_idx_hr + 1: len_hr - max_idx_hr - 1] = 0
    # pose3_mag[len_hr - max_idx: max_idx] = 0
    sp = np.array(pose3_smooth)
    sp.real = np.abs(pose3_mag) / np.sqrt(1 + (np.square(np.tan(pose3_phase)))) * np.sign(np.cos(pose3_phase))
    sp.imag = sp.real * np.tan(pose3_phase)
    pose3_smooth_ifft = np.fft.ifft(sp)
    less = signal.argrelextrema(pose3_smooth_ifft, np.less)[0]
    diff = np.diff(less)
    len_p = np.min(diff)
    less_idx = np.argwhere(diff == len_p)[0][0]
    ppoint_hr = less[less_idx]
    ppoint = ppoint_hr + match_idx - len_p

    '''match_lr'''
    pose3_smooth = np.fft.fft(pose_diff_lr)
    pose3_mag = np.abs(pose3_smooth)
    pose3_phase = np.angle(pose3_smooth)
    pose3_mag[max_idx_lr + int(0.125*len_lr): int(0.875*len_lr) - max_idx_lr - 1] = 0
    sp = np.array(pose3_smooth)
    sp.real = np.abs(pose3_mag) / np.sqrt(1 + (np.square(np.tan(pose3_phase)))) * np.sign(np.cos(pose3_phase))
    sp.imag = sp.real * np.tan(pose3_phase)
    pose3_smooth_ifft = np.fft.ifft(sp)
    less = signal.argrelextrema(pose3_smooth_ifft, np.less)[0]
    ppoints = []
    ppoints.append(ppoint)
    ref = ppoint - len_p
    r = int(len_p/4)
    while (ref >= 0):
        ppoint = np.argwhere(pose3_smooth_ifft == pose3_smooth_ifft[(ref - r): (ref + r)].min())[0][0]
        ppoints.append(ppoint)
        ref = ppoint - len_p
        # a = np.abs(less - ref)
        # b = np.argwhere(a == a.min())[0][0]
        # ppoint = less[b]
        # ppoints.append(ppoint)
        # ref = ppoint - len_p

    ppoint = ppoint_hr + match_idx
    # ppoints.append(ppoint)
    ref = ppoint + len_p
    # print(ref)
    while (ref < len_lr - len_p):
        ppoint = np.argwhere(pose3_smooth_ifft == pose3_smooth_ifft[(ref - r): (ref + r)].min())[0][0]
        ppoints.append(ppoint)
        ref = ppoint + len_p
        # a = np.abs(less - ref)
        # b = np.argwhere(a == a.min())[0][0]
        # ppoint = less[b]
        # ppoints.append(ppoint)
        # ref = ppoint + len_p
    ppoints.sort()
    print(ppoints)
    # x = np.arange(1, pose3_lr.size + 1, 1)
    # # plt.plot(x, pose3_lr, 'r')
    # plt.plot(x[170:208], pose3_smooth_ifft[170:208], 'g')
    # plt.plot(x[194], pose3_smooth_ifft[194], 'r*')
    # # plt.plot(x, pose3_mag)
    # plt.show()

    '''spectral_transfer'''
    pose_output = np.array(pose_lr)
    pose_cur_hr = pose_hr[ppoint_hr: ppoint_hr+len_p, :]
    # x = np.arange(1, len_p + 1, 1)
    for i in range(len(ppoints)-1):
        # f = interp1d(x, pose_cur_hr, kind='cubic')
        len_cur = ppoints[i+1] - ppoints[i]
        # if abs(len_p - len_cur) > (len_p / 5):
        #     continue
        pose_cur = pose_lr[ppoints[i]: ppoints[i+1], :]
        if (len_cur == len_p):
            pose_cur_new = spectral_transfer_p(pose_cur_hr, pose_cur)
            pose_output[ppoints[i]: ppoints[i + 1], :] = pose_cur_new
        else:
            # x_new = np.arange(1, len_cur+1, 1)
            pose_hr_cur_new = resample(pose_cur_hr, len_cur)
            pose_cur_new = spectral_transfer_p(pose_hr_cur_new, pose_cur)
            pose_output[ppoints[i]: ppoints[i + 1], :] = pose_cur_new
    return pose_output



####################################
def mean_smoothing(s,r):
    s2=np.zeros(s.shape)
    len = s.size
    for i in range(r):
        temp1 = 0
        temp2 = 0
        for j in range(i):
            temp1 += s[i]
            temp2 += s[len - i -1]
        s2[i] = temp1 / (i+1)
        s2[len - i - 1] = temp2 / (i+1)
    for i in range(r, len - r):
        tempSum = 0
        for j in range(1, r+1):
            tempSum += (s[i-j] +s[i+j])
        s2[i]=(s[i]+tempSum) / (2*r + 1)
    return s2

def exponential_smoothing(s,alpha,r):
    s2=np.zeros(s.shape)
    len = s.size
    for i in range(r):
        s2[i] = s[i]
        s2[len - i - 1] = s[len - i -1]
    beta = (1-alpha) / (r*2)
    for i in range(r, len - r):
        tempSum = 0
        for j in range(1, r+1):
            tempSum += (s[i-j] +s[i+j])
        s2[i]=alpha*s[i]+beta*tempSum
    return s2

def periodicDecomp(lr, hr, lr_points, hr_points):
    lr = lr
    hr = hr
    lr_num = len(lr_points)-1
    hr_num = len(hr_points)-1
    lr_len = hr_len = 9999
    for i in range(lr_num):
        if lr_points[i+1] - lr_points[i] < lr_len:
            lr_len = lr_points[i+1] - lr_points[i]
    for i in range(hr_num):
        if hr_points[i+1] - hr_points[i] < hr_len:
            hr_len = hr_points[i+1] - hr_points[i]
    if hr_len <= lr_len:
        lr_len = hr_len
    else:
        hr_len = lr_len
    lr_mean = np.mean(lr[lr_points[0]:lr_points[-1]+1], axis=0)
    hr_mean = np.mean(hr[hr_points[0]:hr_points[-1]+1], axis=0)

    results = []
    for k in range(72):
        ### no change rot
        # if k < 3:
        #     results.append(np.array(lr[:, k][lr_points[0]:(lr_points[-1])]))
        #     continue
        hr_4 = hr[:,k] #here
        # hr_pSeg = [6,21,36,51, 67,82]
        hr_pSeg = hr_points
        hr_4_s = []
        hr_segLen = []
        for p in range(hr_num):
            hr_4_s.append(hr_4[hr_pSeg[p]: hr_pSeg[p+1]])
            hr_segLen.append((hr_pSeg[p+1]-hr_pSeg[p])/lr_len)
        hr_part_mean = []
        for j in range(1,hr_len+1):
            tempSum = 0
            tempLen = 0
            for i in range(hr_num):
                tempSum += np.mean(hr_4_s[i][int(hr_segLen[i]*(j-1)):int(hr_segLen[i]*j+1)])
            hr_part_mean.append(tempSum / hr_num)
        hr_factor_mul_4 = np.array(hr_part_mean) / hr_mean[k]
        hr_factor_add_4 = np.array(hr_part_mean) - hr_mean[k]

        lr_4 = lr[:,k] # here
        # lr_pSeg = [0,13,31,47,61,75,90]
        lr_pSeg = lr_points
        lr_4_s = []
        lr_segLen = []
        for i in range(len(lr_pSeg) - 1):
            lr_4_s.append(lr_4[lr_pSeg[i]:lr_pSeg[i+1]])
            lr_segLen.append((lr_pSeg[i+1] - lr_pSeg[i])/lr_len)
        lr_part_mean = []
        for j in range(1,lr_len+1):
            tempSum = 0
            tempLen = 0
            for i in range(lr_num):
                tempSum += np.mean(lr_4_s[i][int(lr_segLen[i]*(j-1)):int(lr_segLen[i]*j+1)])
            lr_part_mean.append(tempSum/lr_num)
        lr_factor_mul_4 = np.array(lr_part_mean) / lr_mean[k]
        lr_factor_add_4 = np.array(lr_part_mean) - lr_mean[k]
        # print(lr_mean[k])

        mline = np.ones([len(lr), 1]) *  lr_mean[k]
        lr_4_m = []
        for i in range(len(lr_pSeg) - 1):
            lr_4_m.append(mline[lr_pSeg[i]:lr_pSeg[i+1]])
        for j in range(1,lr_len+1):
            tempSum = 0
            tempLen = 0
            for i in range(lr_num):
                # print("--j:--",j,"--i:--",i)
                # print(lr_4_s[i][int(segLen[i]*(j-1)):int(segLen[i]*j)])
                lr_4_m[i][int(lr_segLen[i] * (j - 1)):int(lr_segLen[i] * j)] += hr_factor_add_4[j-1]
        result = []
        for i in range(len(lr_4_m)):
            for j in lr_4_m[i]:
                # print(j[0])
                result.append(j[0])
        results.append(np.array(result))
    output = np.array(results).T
    # data = pd.DataFrame(output)
    # data.to_csv('tianyi_pose_0111.csv',header = False, index = False) # here
    # data.to_csv(output_file,header = False, index = False) # here
    return output

def get_hr_points(pose_hr):
    len_hr = pose_hr.shape[0]
    pose_diff_hr = pose_hr[:, 3] - pose_hr[:, 6]
    pose_diff_hr = pose_diff_hr - np.mean(pose_diff_hr)
    max_idx_hr = max_magnitude(pose_hr[:, 3])

    '''find_period'''
    pose3_smooth = np.fft.fft(pose_diff_hr)
    pose3_mag = np.abs(pose3_smooth)
    pose3_phase = np.angle(pose3_smooth)
    pose3_mag[max_idx_hr + 1: len_hr - max_idx_hr - 1] = 0
    # pose3_mag[len_hr - max_idx: max_idx] = 0
    sp = np.array(pose3_smooth)
    sp.real = np.abs(pose3_mag) / np.sqrt(1 + (np.square(np.tan(pose3_phase)))) * np.sign(np.cos(pose3_phase))
    sp.imag = sp.real * np.tan(pose3_phase)
    pose3_smooth_ifft = np.fft.ifft(sp)
    less = signal.argrelextrema(pose3_smooth_ifft, np.less)[0]
    # diff = np.diff(less)
    # len_p = np.min(diff)
    # less_idx = np.argwhere(diff == len_p)[0][0]
    return less



def get_lr_points(pose_lr, pose_hr, match_idx):
    '''cut parts'''
    len_hr = pose_hr.shape[0]
    len_lr = pose_lr.shape[0]
    fore_num = int((match_idx) / len_hr)
    back_num = int((len_lr - match_idx) / len_hr)
    first_end = match_idx - fore_num*len_hr
    last_begin = match_idx + back_num*len_hr

    '''pre'''
    input = np.array(pose_lr)
    pose3_hr = pose_hr[:, 3] - np.mean(pose_hr[:, 3])
    pose6_hr = pose_hr[:, 6] - np.mean(pose_hr[:, 6])
    pose_diff_hr = pose_hr[:, 3] - pose_hr[:, 6]
    pose_diff_hr = pose_diff_hr - np.mean(pose_diff_hr)
    pose3_lr = pose_lr[:, 3] - np.mean(pose_lr[:, 3])
    pose_diff_lr = pose_lr[:, 3] - pose_lr[:, 6]
    pose_diff_lr = pose_diff_lr - np.mean(pose_diff_lr)
    max_idx_hr = max_magnitude(pose_hr[:, 3])
    max_idx_lr = max_magnitude(pose_lr[:, 3])

    '''find_period'''
    pose3_smooth = np.fft.fft(pose_diff_hr)
    pose3_mag = np.abs(pose3_smooth)
    pose3_phase = np.angle(pose3_smooth)
    pose3_mag[max_idx_hr + 1: len_hr - max_idx_hr - 1] = 0
    # pose3_mag[len_hr - max_idx: max_idx] = 0
    sp = np.array(pose3_smooth)
    sp.real = np.abs(pose3_mag) / np.sqrt(1 + (np.square(np.tan(pose3_phase)))) * np.sign(np.cos(pose3_phase))
    sp.imag = sp.real * np.tan(pose3_phase)
    pose3_smooth_ifft = np.fft.ifft(sp)
    less = signal.argrelextrema(pose3_smooth_ifft, np.less)[0]
    diff = np.diff(less)
    len_p = np.min(diff)
    less_idx = np.argwhere(diff == len_p)[0][0]
    ppoint_hr = less[less_idx]
    ppoint = ppoint_hr + match_idx - len_p

    '''match_lr'''
    pose3_smooth = np.fft.fft(pose_diff_lr)
    pose3_mag = np.abs(pose3_smooth)
    pose3_phase = np.angle(pose3_smooth)
    pose3_mag[max_idx_lr + int(0.125*len_lr): int(0.875*len_lr) - max_idx_lr - 1] = 0
    sp = np.array(pose3_smooth)
    sp.real = np.abs(pose3_mag) / np.sqrt(1 + (np.square(np.tan(pose3_phase)))) * np.sign(np.cos(pose3_phase))
    sp.imag = sp.real * np.tan(pose3_phase)
    pose3_smooth_ifft = np.fft.ifft(sp)
    less = signal.argrelextrema(pose3_smooth_ifft, np.less)[0]
    ppoints = []
    ppoints.append(ppoint)
    ref = ppoint - len_p
    r = int(len_p/4)
    while (ref >= 0):
        if (ref < r):
            ppoint = np.argwhere(pose3_smooth_ifft == pose3_smooth_ifft[: (ref + r)].min())[0][0]
            ppoints.append(ppoint)
            ref = ppoint - len_p
            break
        ppoint = np.argwhere(pose3_smooth_ifft == pose3_smooth_ifft[(ref - r): (ref + r)].min())[0][0]
        ppoints.append(ppoint)
        ref = ppoint - len_p

    ppoint = ppoint_hr + match_idx
    # ppoints.append(ppoint)
    ref = ppoint + len_p
    # print(ref)
    while (ref < len_lr - len_p):
        ppoint = np.argwhere(pose3_smooth_ifft == pose3_smooth_ifft[(ref - r): (ref + r)].min())[0][0]
        ppoints.append(ppoint)
        ref = ppoint + len_p

    ppoints.sort()
    return ppoints