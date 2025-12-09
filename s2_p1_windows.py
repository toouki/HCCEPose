# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

'''
s2_p1_gen_pbr_data.py is used to generate PBR data.  
The original script is adapted from BlenderProc2.  
Project link: https://github.com/DLR-RM/BlenderProc

Usage:
    cd HCCEPose
    chmod +x s2_p1_gen_pbr_data.sh
    
    cc0textures:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

    cc0textures-512:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures-512 xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

    
Arguments (example: s2_p1_gen_pbr_data.sh 0 42 ... ):
    Arg 1 (`GPU_ID`): GPU index. Set to 0 for the first GPU.
    Arg 2 (`SCENE_NUM`): Number of scenes; total images generated = 1000 * 42.
    Arg 3 (`cc0textures`): Path to the cc0textures material library.
    Arg 4 (`dataset_path`): Path to the dataset.
    Arg 4 (`s2_p1_gen_pbr_data`): Path to the s2_p1_gen_pbr_data.py script.
    
------------------------------------------------------    

s2_p1_gen_pbr_data.py 用于生成 PBR 数据，原始脚本改编自 BlenderProc2。  
项目链接: https://github.com/DLR-RM/BlenderProc

运行方法:
    cd HCCEPose
    chmod +x s2_p1_gen_pbr_data.sh
    
    cc0textures:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

    cc0textures-512:
    nohup ./s2_p1_gen_pbr_data.sh 0 42 xxx/xxx/cc0textures-512 xxx/xxx/demo-bin-picking xxx/xxx/s2_p1_gen_pbr_data.py > s2_p1_gen_pbr_data.log 2>&1 &

参数说明 (以 s2_p1_gen_pbr_data.sh 0 42 ... 为例):
    参数 1 (`GPU_ID`): GPU 的编号。设置为 0 表示使用第一块显卡。
    参数 2 (`SCENE_NUM`): 场景数量，对应生成的图像数 = 1000 * 42
    参数 3 (`cc0textures`): cc0textures 材质库的路径。
    参数 3 (`dataset_path`): 数据集的路径。
    参数 3 (`s2_p1_gen_pbr_data`): s2_p1_gen_pbr_data.py的路径。
'''

# 导入必要的Python库
import os                    # 操作系统接口，用于文件路径操作
import bpy                   # Blender的Python API，用于3D场景操作
import argparse              # 命令行参数解析库
import blenderproc as bproc  # BlenderProc库，用于3D渲染和数据生成
import numpy as np           # 数值计算库，用于数组操作和数学运算
from tqdm import tqdm        # 进度条显示库
from kasal.utils.io_json import load_json2dict, write_dict2json  # JSON文件读写工具


def set_material_properties(mat, material_type="metal"):
    """
    设置物体的PBR材质属性 - 基于真实物理参数
    
    Args:
        mat: Blender材质对象
        material_type: 材质类型，可选值：
            - "metal": 金属材质
            - "plastic": 塑料材质  
            - "wood": 木材材质
            - "random": 随机材质
    """
    if material_type == "metal":
        # 金属材质设置 - 基于真实金属物理特性
        mat.set_principled_shader_value("Metallic", 1.0)                    # 完全金属
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.05, 0.3))  # 真实金属粗糙度范围
        mat.set_principled_shader_value("Specular", 0.5)                    # 标准镜面反射
        mat.set_principled_shader_value("Base Color", np.random.uniform([0.6, 0.6, 0.6, 1.0], [0.9, 0.9, 0.9, 1.0]))  # 金属基础颜色
        mat.set_principled_shader_value("Anisotropic", np.random.uniform(0.2, 0.6))  # 真实金属各向异性
        mat.set_principled_shader_value("Clearcoat", np.random.uniform(0.0, 0.2))    # 金属清漆层较薄
        mat.set_principled_shader_value("Sheen", 0.0)                      # 金属无光泽层
        mat.set_principled_shader_value("Subsurface", 0.0)                  # 金属无次表面散射
        
    elif material_type == "plastic":
        # 塑料材质设置 - 基于真实塑料物理特性
        mat.set_principled_shader_value("Metallic", 0.0)                    # 非金属
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.3, 0.8))  # 塑料粗糙度范围
        mat.set_principled_shader_value("Specular", np.random.uniform(0.3, 0.6))  # 塑料镜面反射
        # 真实塑料颜色范围
        plastic_colors = [
            [0.95, 0.95, 0.95, 1.0],  # 白色
            [0.9, 0.9, 0.9, 1.0],     # 浅灰
            [0.8, 0.8, 0.8, 1.0],     # 中灰
            [0.95, 0.1, 0.1, 1.0],    # 红色
            [0.1, 0.1, 0.95, 1.0],    # 蓝色
            [0.1, 0.95, 0.1, 1.0],    # 绿色
            [0.95, 0.95, 0.1, 1.0]    # 黄色
        ]
        mat.set_principled_shader_value("Base Color", plastic_colors[np.random.randint(0, len(plastic_colors))])
        mat.set_principled_shader_value("Subsurface", np.random.uniform(0.05, 0.2))  # 塑料次表面散射
        mat.set_principled_shader_value("Anisotropic", 0.0)                  # 塑料无各向异性
        mat.set_principled_shader_value("Clearcoat", np.random.uniform(0.1, 0.4))    # 塑料清漆层
        mat.set_principled_shader_value("Sheen", np.random.uniform(0.0, 0.1))        # 塑料光泽层
        
    elif material_type == "wood":
        # 木材材质设置 - 基于真实木材物理特性
        mat.set_principled_shader_value("Metallic", 0.0)                    # 非金属
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.6, 0.95))  # 木材高粗糙度
        mat.set_principled_shader_value("Specular", np.random.uniform(0.02, 0.15))  # 木材低镜面反射
        # 真实木材颜色范围
        wood_colors = [
            [0.4, 0.3, 0.2, 1.0],     # 深棕色
            [0.6, 0.4, 0.2, 1.0],     # 中棕色
            [0.7, 0.5, 0.3, 1.0],     # 浅棕色
            [0.5, 0.4, 0.3, 1.0],     # 红木色
            [0.3, 0.2, 0.1, 1.0]      # 深色木材
        ]
        mat.set_principled_shader_value("Base Color", wood_colors[np.random.randint(0, len(wood_colors))])
        mat.set_principled_shader_value("Sheen", np.random.uniform(0.2, 0.4))     # 木材光泽层
        mat.set_principled_shader_value("Subsurface", np.random.uniform(0.1, 0.3))  # 木材次表面散射
        mat.set_principled_shader_value("Anisotropic", 0.0)                  # 木材无各向异性
        mat.set_principled_shader_value("Clearcoat", np.random.uniform(0.0, 0.1))    # 木材清漆层较薄
        
    elif material_type == "random":
        # 随机材质设置 - 基于真实物理参数的随机组合
        mat.set_principled_shader_value("Metallic", np.random.choice([0.0, 1.0], p=[0.7, 0.3]))  # 70%非金属，30%金属
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.05, 0.95))
        mat.set_principled_shader_value("Specular", np.random.uniform(0.02, 0.8))
        mat.set_principled_shader_value("Base Color", np.random.uniform([0.1, 0.1, 0.1, 1.0], [0.95, 0.95, 0.95, 1.0]))
        mat.set_principled_shader_value("Anisotropic", np.random.uniform(0.0, 0.6))
        mat.set_principled_shader_value("Clearcoat", np.random.uniform(0.0, 0.4))
        mat.set_principled_shader_value("Sheen", np.random.uniform(0.0, 0.3))
        mat.set_principled_shader_value("Subsurface", np.random.uniform(0.0, 0.3))

def set_metal_type(mat, metal_type="steel"):
    """
    设置特定类型的金属材质 - 基于真实金属物理特性
    
    Args:
        mat: Blender材质对象
        metal_type: 金属类型，可选值：
            - "steel": 钢铁
            - "aluminum": 铝
            - "copper": 铜
            - "gold": 黄金
            - "silver": 银色
    """
    mat.set_principled_shader_value("Metallic", 1.0)  # 完全金属
    mat.set_principled_shader_value("Sheen", 0.0)     # 金属无光泽层
    mat.set_principled_shader_value("Subsurface", 0.0) # 金属无次表面散射
    
    if metal_type == "steel":
        # 钢铁材质 - 基于真实钢铁特性
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.1, 0.4))  # 钢铁粗糙度范围
        mat.set_principled_shader_value("Base Color", [0.65, 0.65, 0.65, 1.0])     # 钢铁灰白色
        mat.set_principled_shader_value("Specular", 0.5)                           # 钢铁镜面反射
        mat.set_principled_shader_value("Anisotropic", np.random.uniform(0.3, 0.6)) # 钢铁各向异性较强
        mat.set_principled_shader_value("Clearcoat", np.random.uniform(0.0, 0.1))   # 钢铁清漆层很薄
        
    elif metal_type == "aluminum":
        # 铝材质 - 基于真实铝特性
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.05, 0.2))  # 铝较光滑
        mat.set_principled_shader_value("Base Color", [0.85, 0.85, 0.85, 1.0])     # 铝银白色
        mat.set_principled_shader_value("Specular", 0.6)                           # 铝镜面反射较强
        mat.set_principled_shader_value("Anisotropic", np.random.uniform(0.1, 0.3)) # 铝各向异性较弱
        mat.set_principled_shader_value("Clearcoat", np.random.uniform(0.0, 0.05))   # 铝清漆层极薄
        
    elif metal_type == "copper":
        # 铜材质 - 基于真实铜特性
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.15, 0.35))  # 铜粗糙度
        mat.set_principled_shader_value("Base Color", [0.72, 0.45, 0.20, 1.0])     # 铜色
        mat.set_principled_shader_value("Specular", 0.5)                           # 铜镜面反射
        mat.set_principled_shader_value("Anisotropic", np.random.uniform(0.2, 0.4)) # 铜各向异性
        mat.set_principled_shader_value("Clearcoat", np.random.uniform(0.0, 0.1))   # 铜清漆层较薄
        
    elif metal_type == "gold":
        # 黄金材质 - 基于真实黄金特性
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.08, 0.25))  # 黄金较光滑
        mat.set_principled_shader_value("Base Color", [1.0, 0.71, 0.29, 1.0])       # 金色
        mat.set_principled_shader_value("Specular", 0.7)                           # 黄金镜面反射很强
        mat.set_principled_shader_value("Anisotropic", np.random.uniform(0.1, 0.3)) # 黄金各向异性
        mat.set_principled_shader_value("Clearcoat", np.random.uniform(0.0, 0.05))  # 黄金清漆层极薄
        
    elif metal_type == "silver":
        # 银材质 - 基于真实银特性
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.06, 0.18))  # 银非常光滑
        mat.set_principled_shader_value("Base Color", [0.75, 0.75, 0.75, 1.0])     # 银色
        mat.set_principled_shader_value("Specular", 0.8)                           # 银镜面反射最强
        mat.set_principled_shader_value("Anisotropic", np.random.uniform(0.05, 0.2)) # 银各向异性很弱
        mat.set_principled_shader_value("Clearcoat", np.random.uniform(0.0, 0.02))  # 银清漆层几乎无


if __name__ == '__main__':
    
    # ==================== 命令行参数解析 ====================
    # Retrieve the GPU ID and other parameters.
    # 获取 GPU 编号和其他参数。
    parser = argparse.ArgumentParser(description='Generate PBR data using BlenderProc')
    # GPU设备ID参数：指定使用哪块GPU进行渲染
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU index (default: 0)')
    # cc0textures材质库路径参数：PBR材质资源库的路径
    parser.add_argument('--cc0textures', type=str, default='cc0textures-512/cc0textures', help='Path to cc0textures material library')
    # 数据集路径参数：包含3D模型的数据集目录路径
    parser.add_argument('--dataset_path', type=str, default='demo-bin-picking2', help='Path to the dataset directory')
    # 场景数量参数：要生成的场景总数，每个场景会渲染20帧图像
    parser.add_argument('--num_scenes', type=int, default=1, help='Number of scenes to generate (default: 50)')
    # 材质类型参数：指定物体的材质类型
    parser.add_argument('--material_type', type=str, default='steel', 
                    choices=['metal', 'plastic', 'wood', 'random', 'random_metal', 'steel', 'aluminum', 'copper', 'gold', 'silver', 'default'],
                    help='Material type for objects')
    args = parser.parse_args()
    gpu_id = args.gpu_id

    # ==================== 数据集路径和配置处理 ====================
    # Use the provided dataset path
    # 使用提供的数据集路径
    current_dir = os.path.abspath(args.dataset_path)      # 获取数据集的绝对路径
    dataset_name = os.path.basename(current_dir)          # 提取数据集名称（最后一级目录名）
    bop_parent_path = os.path.dirname(current_dir)        # 获取数据集的父级目录路径

    # Load the 3D model information of the dataset.
    # 加载数据集的 3D 模型信息。
    # models_info.json包含每个模型的详细信息，如尺寸、类别等
    models_info = load_json2dict(os.path.join(current_dir, 'models', 'models_info.json'))

    # ==================== 相机内参配置 ====================
    # 如果相机配置文件不存在，则创建默认的相机内参配置
    if not os.path.exists(os.path.join(current_dir, 'camera.json')):
        write_dict2json(os.path.join(current_dir, 'camera.json'), 
                            {
                            "cx": 325.2611083984375,      # 主点x坐标（像素）
                            "cy": 242.04899588216654,     # 主点y坐标（像素）
                            "depth_scale": 0.1,            # 深度缩放因子（毫米到米）
                            "fx": 572.411363389757,       # x方向焦距（像素）
                            "fy": 573.5704328585578,      # y方向焦距（像素）
                            "height": 480,                 # 图像高度（像素）
                            "width": 640                   # 图像宽度（像素）
                            }
                        )
    
    # ==================== 模型ID列表提取 ====================
    # Retrieve the list of 3D model IDs from the dataset.
    # 获取数据集中 3D 模型的 ID 列表。
    models_ids = []
    for key in models_info:
        models_ids.append(int(key))    # 将字符串键转换为整数ID
    models_ids = np.array(models_ids)  # 转换为numpy数组以便后续处理

    # ==================== 路径信息打印 ====================
    # Print the parent path and name of the dataset.
    # 打印数据集的父级路径和名称，用于调试和确认路径正确性
    print('-*' * 10)
    print('-*' * 10)
    print('bop_parent_path', bop_parent_path)  # 打印数据集父级路径
    print('dataset_name', dataset_name)        # 打印数据集名称
    print('-*' * 10)
    print('-*' * 10)

    # ==================== 材质库和数据集路径设置 ====================
    # Retrieve the path to the cc0textures assets.
    # 获取 cc0textures 的路径，这是PBR材质资源库
    cc_textures_path = args.cc0textures

    # 构建完整的BOP数据集路径
    bop_dataset_path = os.path.join(bop_parent_path, dataset_name)
    num_scenes = args.num_scenes  # 要生成的场景数量

    # ==================== Blender场景初始化 ====================
    # Create the rendering scene.
    # 创建渲染场景，初始化BlenderProc环境
    bproc.init()  # 初始化BlenderProc，设置Blender环境
    
    # 加载BOP数据集的相机内参
    bproc.loader.load_bop_intrinsics(bop_dataset_path = bop_dataset_path)
    
    # ==================== 3D模型预加载 ====================
    # Load all available objects once outside the loop
    # 在循环外一次性加载所有可用物体，提高效率
    # mm2m=True: 将毫米单位转换为米单位
    all_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = bop_dataset_path, 
                                            mm2m = True,                    # 单位转换：毫米→米
                                            obj_ids = models_ids.tolist()   # 指定要加载的模型ID列表
                                            )
    
    # ==================== 场景环境搭建 ====================
    # 创建房间的五个墙面（立方体房间的5个面，缺少底面让物体可以放置）
    room_planes = [
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),  # 底面（作为地面）
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),  # 前面
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),   # 后面
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),  # 右面
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])  # 左面
    ]
    
    # 为每个墙面设置物理属性（静态刚体，作为碰撞边界）
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
    
    # 创建光源平面（位于场景顶部，作为面光源）
    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    
    # 创建点光源
    light_point = bproc.types.Light()
    light_point.set_energy(100)  # 设置光源强度（提高亮度）

    # ==================== PBR材质加载 ====================
    # Load all texture images from the cc_textures directory.
    # 加载 cc_textures 目录中的所有纹理图。
    # 注释掉的代码：根据纹理库类型选择不同的加载方法
    # if os.path.basename(cc_textures_path) == 'cc0textures-512':
    #     cc_textures = bproc.loader.load_512_ccmaterials(cc_textures_path, use_all_materials=True)
    # else:
    #     cc_textures = bproc.loader.load_ccmaterials(cc_textures_path, use_all_materials=True)
    # 直接加载512x512分辨率的PBR材质库
    cc_textures = bproc.loader.load_512_ccmaterials(cc_textures_path, use_all_materials=True)
    
    # ==================== 物体位姿采样函数定义 ====================
    def sample_pose_func(obj: bproc.types.MeshObject):
        """
        为物体随机采样位姿（位置和姿态）
        Args:
            obj: 要设置位姿的物体对象
        """
        # 随机生成位置的最小值范围（x, y, z坐标）
        min = np.random.uniform([-0.15, -0.15, 0.0], [-0.1, -0.1, 0.0])
        # 随机生成位置的最大值范围（x, y, z坐标）
        max = np.random.uniform([0.1, 0.1, 0.4], [0.15, 0.15, 0.6])
        # 在指定范围内随机设置物体位置
        obj.set_location(np.random.uniform(min, max))
        # 随机设置物体的旋转姿态（使用SO3均匀采样）
        obj.set_rotation_euler(bproc.sampler.uniformSO3())

    # ==================== 渲染器配置 ====================
    # 启用深度图输出，关闭抗锯齿以提高渲染速度
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    # 提高采样样本数以获得更好的金属反射效果
    bproc.renderer.set_max_amount_of_samples(100)
    
    # ==================== 光追渲染配置 ====================
    # 启用光追加速（需要支持RTX的NVIDIA显卡）
    bproc.renderer.set_denoiser('OPTIX')   # 使用NVIDIA OptiX降噪器
    bproc.renderer.set_noise_threshold(0.005)  # 降低噪声阈值，提高渲染质量
    bproc.renderer.set_light_bounces(max_bounces=12, glossy_bounces=8, transmission_bounces=8)  # 增加光线反弹次数

    # ==================== GPU设备设置 ====================
    # Set the GPU ID.
    # 设置 GPU 的编号，指定使用哪块GPU进行渲染
    bproc.renderer.set_render_devices(desired_gpu_device_type='CUDA', desired_gpu_ids = [gpu_id])

    # ==================== 主循环：场景生成 ====================
    # 对每个场景进行渲染处理
    for i in tqdm(range(num_scenes)):
        
        # ==================== 物体选择策略 ====================
        # Bin-picking selection mode.
        # bin-picking 的挑选模式（模拟工业抓取场景中的物体堆叠）
        
        idx_l = np.random.choice(models_ids, size=1, replace=True)  # 随机选择2个物体ID
        obj_ids = []
        for _ in range(15):
            obj_ids.append(int(idx_l[0]))  # 第一种物体15个实例
            # obj_ids.append(int(idx_l[1]))  # 第二种物体15个实例
        
        # Multi-class object picking mode.
        # 多类别物体的挑选模式。（当前被注释掉，未启用）
        # rand_s = np.random.rand()  # 生成随机数，用于选择不同的物体挑选模式（当前未使用）
        # if rand_s > 0.5:
        #     idx_l = np.random.choice(models_ids, size=30, replace=True)  # 有放回随机选择30个物体
        # else:
        #     idx_l = np.random.choice(models_ids, size=min(models_ids.shape[0],30), replace=False)  # 无放回随机选择
        # obj_ids = []
        # for idx_i in idx_l:
        #     obj_ids.append(int(idx_i))
        
        # ==================== 物体实例化处理 ====================
        # Select objects from pre-loaded objects
        # 从预加载的物体中选择并创建实例
        sampled_target_bop_objs = []
        
        # 统计每个物体ID需要的实例数量
        obj_id_counts = {}
        for obj_id in obj_ids:
            obj_id_counts[obj_id] = obj_id_counts.get(obj_id, 0) + 1
        
        # 根据统计的数量创建物体实例
        for obj_id, count in obj_id_counts.items():
            # Find the original object for this ID
            # 在预加载的物体中查找对应ID的原始物体
            original_objs = [obj for obj in all_bop_objs if obj.get_cp("category_id") == obj_id]
            if original_objs:
                original_obj = original_objs[0]  # 获取第一个匹配的物体
                # Create the required number of instances
                # 创建所需数量的物体实例
                for i in range(count):
                    if i == 0:
                        # Use the original object for the first instance
                        # 第一个实例直接使用原始物体
                        sampled_target_bop_objs.append(original_obj)
                    else:
                        # Duplicate the object for additional instances
                        # 额外的实例通过复制原始物体创建
                        duplicated_obj = original_obj.duplicate()
                        sampled_target_bop_objs.append(duplicated_obj)
        
        # ==================== 物体材质和物理属性配置 ====================
        # Set object materials and poses, then render 20 frames.
        # 设置物体的材质和位姿，并渲染 20 帧图像。
        
        # Hide all objects first
        # 首先隐藏所有物体，避免干扰当前场景
        for obj in all_bop_objs:
            obj.hide(True)              # 隐藏物体
            obj.disable_rigidbody()     # 禁用刚体物理属性
        
        # Show and configure selected objects
        # 显示并配置选中的物体
        for obj in sampled_target_bop_objs:
            obj.set_shading_mode('auto')  # 设置自动着色模式
            mat = obj.get_materials()[0]  # 获取物体的第一个材质
            
            # ==================== 根据命令行参数设置材质 ====================
            # 根据用户指定的材质类型进行设置
            material_type = args.material_type
            
            if material_type in ['metal', 'plastic', 'wood', 'random']:
                # 使用通用材质函数
                set_material_properties(mat, material_type=material_type)
            elif material_type == 'random_metal':
                # 随机选择金属类型（默认选项）
                metal_types = ['steel', 'aluminum', 'copper', 'gold', 'silver']
                random_metal = np.random.choice(metal_types)
                set_metal_type(mat, metal_type=random_metal)
            elif material_type in ['steel', 'aluminum', 'copper', 'gold', 'silver']:
                # 使用特定金属类型
                set_metal_type(mat, metal_type=material_type)
            else:
                # 随机设置PBR材质参数
                mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))    # 粗糙度：0(光滑)到1(粗糙)
                mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))      # 镜面反射强度
            # 启用刚体物理属性，用于物理模拟
            obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
            obj.hide(False)  # 显示物体
        
        # ==================== 光照配置 ====================
        # 配置面光源的发光属性
        light_plane_material.make_emissive(emission_strength=np.random.uniform(10,40),    
                                        emission_color=np.random.uniform([0.8, 0.8, 0.8, 1.0], [1.0, 1.0, 1.0, 1.0]))  # 发光颜色（提高亮度）
        light_plane.replace_materials(light_plane_material)
        
        # 配置点光源的颜色和位置
        light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))  # 随机光源颜色（灰白色到白色）
        # 在球形壳上随机采样点光源位置
        location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                                elevation_min = 20, elevation_max = 89)  # 仰角范围：5°到89°
        light_point.set_location(location)
        
        # ==================== 环境纹理配置 ====================
        # 随机选择一个PBR材质应用到所有墙面
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)  # 为每个墙面应用相同的随机材质

        # ==================== 物体位姿采样和物理模拟 ====================
        # 为所有选中的物体随机采样初始位姿
        bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs,
                                sample_pose_func = sample_pose_func,     # 使用前面定义的位姿采样函数
                                max_tries = 1000)                        # 最大尝试次数，避免无限循环
                
        # 进行物理模拟，让物体自然下落并堆叠
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,   # 最小模拟时间（秒）
                                                        max_simulation_time=10,  # 最大模拟时间（秒）
                                                        check_object_interval=1,  # 检查物体状态的时间间隔
                                                        substeps_per_frame = 20,  # 每帧的物理子步数
                                                        solver_iters=25)          # 物理求解器迭代次数

        # ==================== 相机位姿生成 ====================
        # 创建所有物体的BVH树，用于碰撞检测和视线检查
        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs)

        # 生成20个相机位姿
        cam_poses = 0
        while cam_poses < 20:
            # 在球形壳上随机采样相机位置
            location = bproc.sampler.shell(center = [0, 0, 0],    # 场景中心
                                    radius_min = 0.5,            # 最小距离
                                    radius_max = 2,            # 最大距离
                                    elevation_min = 20,           # 最小仰角20°
                                    elevation_max = 89)          # 最大仰角89°
            
            # 计算兴趣点（Point of Interest），相机将朝向这个点
            # Fix sampling error: ensure we don't try to sample more objects than available
            sample_size = min(len(sampled_target_bop_objs), int(round(0.6 * len(obj_ids))))  # 采样物体数量的60%
            if sample_size > 0:
                poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=sample_size, replace=False))
            else:
                poi = np.array([0, 0, 0])  # Default POI if no objects
            
            # 根据相机位置和兴趣点计算相机旋转矩阵
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
            # 构建相机的变换矩阵（位置+旋转）
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            
            # 检查相机视线上是否有障碍物（确保相机能看到物体）
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
                bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)  # 添加相机位姿
                cam_poses += 1
        # ==================== 渲染和数据输出 ====================
        # 执行渲染，获取RGB图像和深度图
        data = bproc.renderer.render()
        
        # Create unique output directory for each scene to avoid overwriting
        # scene_output_dir = os.path.join(f'train_pbr/',f"scene_{i:06d}")
        
        # 将渲染结果保存为BOP格式的数据集
        bproc.writer.write_bop(bop_parent_path,                    # BOP数据集的父目录
                            target_objects = sampled_target_bop_objs,  # 目标物体列表
                            dataset = dataset_name,                    # 数据集名称
                            depth_scale = 0.1,                         # 深度缩放因子（毫米到米）
                            depths = data["depth"],                    # 深度图数据
                            colors = data["colors"],                   # RGB图像数据
                            color_file_format = "JPEG",               # 图像文件格式
                            ignore_dist_thres = 10,                    # 忽略距离阈值（米）
                            append_to_existing_output = True)          # 追加到现有输出
        
        # ==================== 场景清理 ====================
        # 清理当前场景的物体，为下一个场景做准备
        for obj in (sampled_target_bop_objs):    
            obj.disable_rigidbody()  # 禁用刚体属性
            obj.hide(True)           # 隐藏物体
    
    pass  # 程序结束