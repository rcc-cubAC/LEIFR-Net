
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import face_alignment
import os
import face_alignment
import imageio
import numpy as np

def process_images(input_folder, output_folder):
    # 创建输出文件夹，如果不存在的话
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 初始化face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # 检查文件是否为图片
            # 读取图片
            input_path = os.path.join(input_folder, filename)
            image = imageio.imread(input_path)

            # 获取面部特征点
            preds = fa.get_landmarks(image)
            
            # 检查是否检测到面部特征点
            if preds is not None or preds[0].shape[0] == 68:
                # 保存面部特征点到.npy文件
                output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".npy")
                np.save(output_path, preds[0]/image.shape[1])  # 如果有多张脸，这里只保存第一张脸的特征点
            else:
                print('error')
if __name__ == "__main__":
    input_folder = 'E:/Relight/dataset/test/face'  # 替换为你的输入文件夹路径
    output_folder = 'E:/Relight/dataset/test/keypoint'  # 替换为你的输出文件夹路径
    process_images(input_folder, output_folder)
