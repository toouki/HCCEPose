import cv2, os, sys
import numpy as np
from HccePose.bop_loader import bop_dataset
from HccePose.tester import Tester

if __name__ == '__main__':
    
    sys.path.insert(0, os.getcwd())
    current_dir = "/mnt/a23f9bec-57bd-461b-8d1d-06d98a9f961c/dzy/HCCEPose-main"
    dataset_path = os.path.join(current_dir, 'demo-bin-picking')
    test_img_path = os.path.join(current_dir, 'test_imgs')
    bop_dataset_item = bop_dataset(dataset_path)
    obj_id = 1
    CUDA_DEVICE = '0'
    # show_op = False
    show_op = True
    
    Tester_item = Tester(bop_dataset_item, show_op = show_op, CUDA_DEVICE=CUDA_DEVICE)
    
    # 获取 test_img_dir 中所有 .jpg 文件
    image_files = [f for f in os.listdir(test_img_path) if f.endswith('.jpg') and '_show_' not in f]
    
    for name in image_files:
        file_name = os.path.join(test_img_path, name)
        if not os.path.exists(file_name):
            print(f"警告: 文件不存在: {file_name}")
            continue
            
        image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_RGB2BGR)
        if image is None:
            print(f"警告: 无法读取图像: {file_name}")
            continue
            
        cam_K = np.array([
            [2.83925618e+03, 0.00000000e+00, 2.02288638e+03],
            [0.00000000e+00, 2.84037288e+03, 1.53940473e+03],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
        ])
        results_dict = Tester_item.predict(cam_K, image, [obj_id],
                                                        conf = 0.5, confidence_threshold = 0.5)
        cv2.imwrite(file_name.replace('.jpg','_show_2d.jpg'), results_dict['show_2D_results'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis0.jpg'), results_dict['show_6D_vis0'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis1.jpg'), results_dict['show_6D_vis1'])
        cv2.imwrite(file_name.replace('.jpg','_show_6d_vis2.jpg'), results_dict['show_6D_vis2'])
        
        # 保存6D位姿为txt文件（同一张图片的所有物体保存到一个文件）
        pose_output_dir = os.path.join(test_img_path, 'poses')
        os.makedirs(pose_output_dir, exist_ok=True)
        
        pose_filename = f"{os.path.splitext(name)[0]}_poses.txt"
        pose_filepath = os.path.join(pose_output_dir, pose_filename)
        
        with open(pose_filepath, 'w') as f:
            f.write(f"# 6D Poses for Image: {name}\n")
            f.write(f"# Camera Matrix K:\n")
            f.write(f"# {cam_K[0,0]:.6f} {cam_K[0,1]:.6f} {cam_K[0,2]:.6f}\n")
            f.write(f"# {cam_K[1,0]:.6f} {cam_K[1,1]:.6f} {cam_K[1,2]:.6f}\n")
            f.write(f"# {cam_K[2,0]:.6f} {cam_K[2,1]:.6f} {cam_K[2,2]:.6f}\n")
            f.write("# Format: obj_id, detection_id, confidence, R(3x3), t(3x1)\n")
            f.write("# ----------------------------------------------------------------------\n")
            
            detection_count = 0
            for detected_obj_id in results_dict:
                if detected_obj_id != 'time' and detected_obj_id not in ['show_2D_results', 'show_6D_vis0', 'show_6D_vis1', 'show_6D_vis2']:
                    pred_results = results_dict[detected_obj_id]
                    if 'Rts' in pred_results:
                        Rts = pred_results['Rts']
                        confs = pred_results['conf'].cpu().numpy()
                        
                        for i, (Rt, conf) in enumerate(zip(Rts, confs)):
                            # 提取旋转矩阵和平移向量
                            R = Rt[:3, :3]
                            t = Rt[:3, 3]
                            
                            # 写入格式：obj_id detection_id confidence R11 R12 R13 R21 R22 R23 R31 R32 R33 t1 t2 t3
                            f.write(f"{int(detected_obj_id):02d} {detection_count:02d} {conf:.6f} ")
                            # 写入旋转矩阵
                            for row in R:
                                f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} ")
                            # 写入平移向量
                            f.write(f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\n")
                            
                            detection_count += 1
        
        print(f"保存位姿文件: {pose_filepath} (共{detection_count}个检测结果)")
