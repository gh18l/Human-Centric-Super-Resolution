import os
import cfg

def openpose_processing(base_path):
    '''
    base_path: the image is in base_path/optimization_data
    result is saved in base_path/optimization_data/COCO or /MPI
    '''
    openpose_path = cfg.openpose_path
    os.system('cd ' + cfg.openpose_path + ' && ./build/examples/openpose/openpose.bin --image_dir ' + os.path.join(base_path, 'optimization_data') + ' --write_images ' + os.path.join(base_path, 'optimization_data/COCO/images') + ' --write_json ' + os.path.join(base_path, 'optimization_data/COCO'))
    os.system('cd ' + cfg.openpose_path + ' && ./build/examples/openpose/openpose.bin --image_dir ' + os.path.join(base_path, 'optimization_data') + ' --model_pose "MPI" --write_images ' + os.path.join(base_path, 'optimization_data/MPI/images') + ' --write_json ' + os.path.join(base_path + 'optimization_data/MPI'))

def PSPNet_processing(base_path):
    PSP_path = cfg.PSP_path
    os.system('cd ' + PSP_path + ' && source activate PSPNet && rm ' + os.path.join(PSP_path, 'images/*') + ' && rm ' + os.path.join(PSP_path, 'results/*') + ' && cp ' + os.path.join(base_path, 'optimization_data/*.png') + ' ' + os.path.join(PSP_path, 'images') + ' && python pspnet.py && cp ' + os.path.join(PSP_path, 'results/*') + ' ' + os.path.join(base_path, optimization_data))

def hmr_processing(base_path):
    hmr_path = cfg.hmr_path
    os.system('cd ' + hmr_path ' && source venv/bin/activate && python demo.py ' + base_path + ' && deactivate')


    