import os
import ujson
import numpy as np
import argparse

def get_samples_info(root, clips):
    clips_info = []
    samples_info = {clip: [] for clip in clips}

    for clip in clips:
        clip_path = os.path.join(root, clip)
        samples = sorted(os.listdir(clip_path), key=lambda x: int(x.split("/")[-1].split("_")[0]))

        clips_info.append({'clip_name': clip,
                                'index': [len(samples), len(samples) + len(samples)]
                                })

        for j in range(len(samples)):
            samples_info[clip].append(os.path.join(clip_path, samples[j]))
    return samples_info

def T_inv_function(matrix):
    # 定义旋转矩阵 R 和平移向量 t
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    # 计算变换矩阵的逆
    T_inv = np.eye(4)
    T_inv[:3, :3] = np.linalg.inv(R)
    T_inv[:3, 3] = -np.linalg.inv(R) @ t

    return T_inv

def main(args):
    clips = sorted(os.listdir(args.root_dir), key=lambda x: int(x.split("_")[1]))
    samples_info = get_samples_info(args.root_dir, clips)

    for key, values in samples_info.items():
        print(f"Values for {key}:")
        save_dir_path = os.path.join(args.root_dir, key + '_reverse')
        # 检查指定路径是否存在
        if not os.path.exists(save_dir_path):
            # 如果不存在，则创建该目录
            os.makedirs(save_dir_path)

        for id, value in enumerate(values):
            with open(value, 'rb') as fp:
                data = ujson.load(fp)

            save_name = values[-1-id].split('/')[-1]


            print('id:', id)
            data_1 = np.array(data["pc1"])
            data_2 = np.array(data["pc2"])
            trans = np.array(data["trans"])
            trans = T_inv_function(trans)
            gt_mask1 = np.array(data["gt_mask1"])
            gt_label1 = np.array(data["gt_label1"])
            gt_mask2 = np.array(data["gt_mask2"])
            gt_label2 = np.array(data["gt_label2"])

            # all info
            sample = {
                "pc1": data_2.tolist(),
                "pc2": data_1.tolist(),
                "trans": trans.tolist(),
                "opt_info": data["opt_info"],
                "gt_mask1": gt_mask2.tolist(),
                "gt_label1": gt_label2.tolist(),
                "gt_mask2": gt_mask1.tolist(),
                "gt_label2": gt_label1.tolist(),
            }
            res_path = os.path.join(save_dir_path, save_name)
            ujson.dump(sample, open(res_path, "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--root_dir', type=str, default="/mnt/12T/fangqiang/view_of_delft/", help='Path for the origial dataset.')
    args = parser.parse_args()
    main(args)