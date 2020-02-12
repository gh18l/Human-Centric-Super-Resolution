import smpl_np
import core
import numpy as np
import scipy.signal as signal
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def render_naked(theta, beta, tran):
    smpl = smpl_np.SMPLModel('./models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    renderer = core.SMPLRenderer(face_path="./models/smpl_faces.npy")
    verts = smpl.get_verts(theta, beta, tran)
    render_result = renderer(verts, cam=None, img=None, do_alpha=False)  ## alpha channel
    return render_result

def render_naked_imgbg(theta, beta, tran, img, camera):
    """
    camera = [focal, cx, cy, trans]
    """
    smpl = smpl_np.SMPLModel('./models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    renderer = core.SMPLRenderer(face_path="./models/smpl_faces.npy")
    verts = smpl.get_verts(theta, beta, tran)
    camera_for_render = np.hstack([camera[0], camera[1], camera[2], camera[3]])
    render_result = renderer(verts, cam=camera_for_render, img=img, do_alpha=False)
    return render_result

def render_naked_rotation(theta, beta, tran, angle):
    smpl = smpl_np.SMPLModel('./models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    renderer = core.SMPLRenderer(face_path="./models/smpl_faces.npy")
    verts = smpl.get_verts(theta, beta, tran)
    render_result = renderer.rotated(verts, angle, cam=None, img_size=None)
    return render_result

def max_magnitude(pose):
    sp = np.fft.fft(pose)
    mag = np.abs(sp)
    mag_short = np.array(mag[1:])
    # print(mag)
    idx = np.argwhere(mag == np.max(mag_short))
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
    ppoint = ppoint_hr + match_idx

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
    while (ref < len_lr):
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



def main():
    '''input parameter'''
    pose_lr = np.load('/media/zyp/ZYP/ubuntu/workplace/data/LRmotion5.28/optimization_pose.npy')
    pose_hr = np.loadtxt('/media/zyp/MEOW/HRmotion5.28/output/optimization_pose.csv', dtype=np.double, delimiter=',')
    match_idx = 485


    pose_output = refineLR(pose_lr, pose_hr, match_idx)



    # 'every part spectral_transfer'
    # if first_end != 0:
    #     pose_input = input[0: first_end, :]
    #     pose_output_first = spectral_transfer(pose_hr[-(first_end):, :], pose_lr[-(first_end):, :], pose_input, max_idx)
    #     result = np.array(pose_output_first)
    # for i in range(1, fore_num+1):
    #     begin_idx = match_idx - i*len_hr
    #     pose_input = input[begin_idx: begin_idx + len_hr, :]
    #     pose_output = spectral_transfer(pose_hr, pose_lr, pose_input, max_idx)
    #     result = np.vstack((result, pose_output))
    # for i in range(back_num):
    #     begin_idx = match_idx + i*len_hr
    #     pose_input = input[begin_idx: begin_idx + len_hr, :]
    #     pose_output = spectral_transfer(pose_hr, pose_lr, pose_input, max_idx)
    #     result = np.vstack((result, pose_output))
    # if last_begin < len_lr:
    #     pose_input = input[last_begin:, :]
    #     pose_output_last = spectral_transfer(pose_hr[0: len_lr-last_begin, :], pose_lr[0: len_lr-last_begin, :], pose_input, max_idx)
    #     result = np.vstack((result, pose_output_last))

    for i in range(pose_lr.shape[0]):
        # theta = np.zeros(72)
        # theta[0] = np.pi
        # theta1 = pose_lr[i,:]
        theta1 = pose_lr[i, :]
        # theta_hr = pose_hr[i,:]
        theta2 = pose_output[i,:]
        beta = np.ones(10) * .03
        tran = np.zeros(3)
        render_result1 = render_naked(theta1, beta, tran)
        # render_result_hr = render_naked(theta_hr, beta, tran)
        render_result2 = render_naked(theta2, beta, tran)
        # render_result = np.hstack((render_result1, render_result_hr, render_result2))
        render_result = np.hstack((render_result1, render_result2))
        cv2.imwrite('./output/052808/{:04d}.png'.format(i),render_result)

if __name__ == '__main__':
    main()