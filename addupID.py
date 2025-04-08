import os
import json
import sys
import argparse # 用于处理命令行参数

def update_speaker_ids(folder_a_path, file_b_path):
    """
    将文件夹A内所有第一层子文件的名称添加到指定JSON文件B的spk2id字典中，
    并更新n_speakers计数。假设JSON文件B必须存在。

    Args:
        folder_a_path (str): 文件夹A的路径。
        file_b_path (str): 文件B (JSON文件) 的路径。
    """
    print(f"--- 开始处理 ---")
    print(f"源文件夹 (Folder A): {folder_a_path}")
    print(f"目标JSON文件 (File B): {file_b_path}")

    # --- 1. 检查并获取文件夹A中的文件名 ---
    if not os.path.isdir(folder_a_path):
        print(f"错误：找不到源文件夹 '{folder_a_path}'")
        sys.exit(1)

    try:
        all_items = os.listdir(folder_a_path)
        # 过滤出第一层的文件
        file_names = [f for f in all_items if os.path.isfile(os.path.join(folder_a_path, f))]
        print(f"在源文件夹中找到 {len(file_names)} 个文件。")
        if not file_names:
            print("源文件夹中没有文件，无需更新。")
            print("--- 处理结束 ---")
            return

    except OSError as e:
        print(f"错误：读取源文件夹时出错: {e}")
        sys.exit(1)

    # --- 2. 读取并解析文件B (JSON) ---
    # 现在假设文件必须存在，如果找不到则报错退出
    data = None
    try:
        with open(file_b_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取JSON文件: {file_b_path}")

        # 验证基本结构
        if not isinstance(data, dict) or \
           "data" not in data or \
           not isinstance(data["data"], dict) or \
           "spk2id" not in data["data"] or \
           not isinstance(data["data"]["spk2id"], dict) or \
           "n_speakers" not in data["data"]:
             raise ValueError("JSON文件结构不符合预期。缺少 'data' 或 'data['spk2id']' 或 'data['n_speakers']'。")

    except FileNotFoundError:
        print(f"错误：找不到指定的JSON配置文件 '{file_b_path}'。请确保文件存在。")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误：JSON文件 '{file_b_path}' 不是有效的JSON格式。请检查文件内容。")
        sys.exit(1)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误：读取或解析JSON文件时发生未知错误: {e}")
        sys.exit(1)

    # --- 3. 更新 spk2id 字典 ---
    spk2id_dict = data["data"]["spk2id"]
    existing_speakers = set(spk2id_dict.keys())
    current_max_id = -1
    if spk2id_dict:
        try:
            current_max_id = max(spk2id_dict.values()) if spk2id_dict else -1
        except ValueError:
            print(f"警告：'spk2id' 字典中的值似乎不是有效的ID（数字）。将从ID 0开始分配。")
            current_max_id = -1 # 如果值无效，则重置

    next_id = current_max_id + 1
    added_count = 0

    print("开始添加新的speaker ID...")
    for filename in sorted(file_names):
        # 使用完整文件名作为key
        key_name = filename
        # # 如果需要去除扩展名，取消下面这行的注释
        # key_name, _ = os.path.splitext(filename)

        if key_name not in existing_speakers:
            spk2id_dict[key_name] = next_id
            print(f"  添加: \"{key_name}\": {next_id}")
            next_id += 1
            added_count += 1
        else:
            print(f"  跳过 (已存在): \"{key_name}\"")

    if added_count > 0:
        print(f"成功添加了 {added_count} 个新的speaker ID。")
    else:
        print("没有新的speaker ID需要添加。")

    # --- 4. 更新 n_speakers 计数 ---
    new_total_speakers = len(spk2id_dict)
    data["data"]["n_speakers"] = new_total_speakers
    print(f"更新 'n_speakers' 为: {new_total_speakers}")

    # --- 5. 写回文件B (JSON) ---
    try:
        with open(file_b_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"成功将更新后的数据写回文件: {file_b_path}")
    except Exception as e:
        print(f"错误：写入JSON文件时出错: {e}")
        sys.exit(1)

    print("--- 处理结束 ---")

# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="更新JSON配置文件中的speaker ID列表。")

    parser.add_argument(
        "-r", "--read-folder",
        default="wavs/", # 设置默认相对路径
        help="包含要读取文件名的源文件夹路径 (必需)。"
    )
    parser.add_argument(
        "-c", "--config",
        default="config/config.json", # 设置默认相对路径
        help="目标JSON配置文件的路径 (默认为: config/config.json)。"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 获取参数值
    folder_path = args.read_folder
    config_path = args.config

    # --- 运行主函数 ---
    update_speaker_ids(folder_path, config_path)
