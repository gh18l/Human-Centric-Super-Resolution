import smpl_np
import core
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    idx = np.argwhere(mag_short == np.max(mag_short))
    return idx[0][0]

'''spectral transfer'''
def spectral_transfer(hr, lr, input, max_idx):
    output = np.array(input)
    len_hr = hr.shape[0]
    per25 = int(len_hr*0.25)
    per125 = int(len_hr*0.125)
    for j in range(3,72):
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
        magnitude[max_idx+1: len_hr - (max_idx+1)] = (magnitude[max_idx+1: len_hr - (max_idx+1)] + mag_d[max_idx+1: len_hr - (max_idx+1)] + magnitude_hr[max_idx+1: len_hr - (max_idx+1)])/2
        magnitude[per25:len_hr-per25] = (magnitude[per25:len_hr-per25] + mag_d[per25:len_hr-per25] + magnitude_hr[per25:len_hr-per25])/4
        magnitude[per125:len_hr-per125] = (magnitude[per125:len_hr-per125] + mag_d[per125:len_hr-per125] + magnitude_hr[per125:len_hr-per125])/8
        phase = np.array(phase_input)
        sp.real = np.abs(magnitude) / np.sqrt(1 + (np.square(np.tan(phase)))) * np.sign(np.cos(phase))
        sp.imag = sp.real * np.tan(phase)
        output[:, j] = np.fft.ifft(sp)
    return output



def main():
    '''input parameter'''
    pose_lr = np.load('/media/zyp/ZYP/ubuntu/workplace/data/motion_refine5.22/optimization_pose.npy')
    pose_hr = np.loadtxt('/media/zyp/ZYP/ubuntu/workplace/data/motion_refine5.22/HR/optimization_pose.csv', dtype=np.double, delimiter=',')
    match_idx = 485

    '''cut parts'''
    len_hr = pose_hr.shape[0]
    len_lr = pose_lr.shape[0]
    fore_num = int((match_idx) / len_hr)
    back_num = int((len_lr - match_idx) / len_hr)
    first_end = match_idx - fore_num*len_hr
    last_begin = match_idx + back_num*len_hr

    '''pre'''
    input = np.array(pose_lr)
    pose3 = pose_hr[:, 3]
    max_idx = max_magnitude(pose3) + 1
    pose_lr = pose_lr[match_idx: match_idx+len_hr, :]

    'every part spectral_transfer'
    if first_end != 0:
        pose_input = input[0: first_end, :]
        pose_output_first = spectral_transfer(pose_hr[-(first_end):, :], pose_lr[-(first_end):, :], pose_input, max_idx)
        result = np.array(pose_output_first)
    for i in range(1, fore_num+1):
        begin_idx = match_idx - i*len_hr
        pose_input = input[begin_idx: begin_idx + len_hr, :]
        pose_output = spectral_transfer(pose_hr, pose_lr, pose_input, max_idx)
        result = np.vstack((result, pose_output))
    for i in range(back_num):
        begin_idx = match_idx + i*len_hr
        pose_input = input[begin_idx: begin_idx + len_hr, :]
        pose_output = spectral_transfer(pose_hr, pose_lr, pose_input, max_idx)
        result = np.vstack((result, pose_output))
    if last_begin < len_lr:
        pose_input = input[last_begin:, :]
        pose_output_last = spectral_transfer(pose_hr[0: len_lr-last_begin, :], pose_lr[0: len_lr-last_begin, :], pose_input, max_idx)
        result = np.vstack((result, pose_output_last))

if __name__ == '__main__':
    main()