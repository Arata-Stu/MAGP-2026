import argparse
import multiprocessing  
import os               
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader


def extract_and_save_per_bag(bag_path, output_dir, image_topic, cmd_topic, odom_topic, scan_topic):
    """
    単一のrosbagファイルからデータを抽出する並列ワーカー関数。
    (image or scan) and control が必須。odom は任意。
    image と scan が両方ある場合は image を同期ベースとする。

    output
    /path/to/dataset/
    ├── seq_01/
    │   ├── images/
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ... (同期されたフレーム数分)
    │   ├── odoms.npy
    │   ├── scans.npy
    │   ├── speeds.npy
    │   └── steers.npy
    │
    └── seq_02/
        ├── images/
        │   ├── 000000.png
        │   ├── 000001.png
        │   └── ...
        ├── odoms.npy
        ├── scans.npy
        ├── speeds.npy
        └── steers.npy
    """
    pid = os.getpid()  
    bag_path = Path(bag_path).expanduser().resolve()
    bag_name = bag_path.name
    out_dir = Path(output_dir) / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image_data, image_times = [], []
    cmd_data, cmd_times = [], []
    odom_data, odom_times = [], []
    scan_data, scan_times = [], []

    try:
        with AnyReader([bag_path]) as reader:
            topic_list = [image_topic, cmd_topic, odom_topic, scan_topic]
            connections = [c for c in reader.connections if c.topic in topic_list]

            for conn, timestamp, raw in reader.messages(connections=connections):
                msg = reader.deserialize(raw, conn.msgtype)

                if conn.topic == image_topic and conn.msgtype == 'sensor_msgs/msg/Image':
                    encoding = msg.encoding
                    if encoding == 'mono8':
                        shape = (msg.height, msg.width)
                    elif encoding in ['bgr8', 'rgb8']:
                        shape = (msg.height, msg.width, 3)
                    else:
                        continue

                    image_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(shape)
                    if encoding == 'rgb8':
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                    image_data.append(image_np)
                    image_times.append(timestamp)

                elif conn.topic == cmd_topic and conn.msgtype == 'ackermann_msgs/msg/AckermannDriveStamped':
                    cmd_vec = np.array([msg.drive.steering_angle, msg.drive.acceleration], dtype=np.float32)
                    cmd_data.append(cmd_vec)
                    cmd_times.append(timestamp)

                elif conn.topic == odom_topic and conn.msgtype == 'nav_msgs/msg/Odometry':
                    pose = msg.pose.pose
                    position = np.array([pose.position.x, pose.position.y, pose.position.z])
                    orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
                    odom_vec = np.concatenate([position, orientation]).astype(np.float32)
                    odom_data.append(odom_vec)
                    odom_times.append(timestamp)
                
                elif conn.topic == scan_topic and conn.msgtype == 'sensor_msgs/msg/LaserScan':
                    scan_vec = np.array(msg.ranges, dtype=np.float32)
                    scan_data.append(scan_vec)
                    scan_times.append(timestamp)


    except Exception as e:
        print(f"[PID:{pid} ERROR] {bag_name}: Failed to read bag file. {e}")
        return

    if len(cmd_data) == 0:
        print(f'[PID:{pid} WARN] Skipping {bag_name}: insufficient data (commands missing)')
        return
    
    if len(image_data) == 0 and len(scan_data) == 0:
        print(f'[PID:{pid} WARN] Skipping {bag_name}: insufficient data (both images and scans missing)')
        return

    has_image = len(image_data) > 0
    has_scan = len(scan_data) > 0
    has_odom = len(odom_data) > 0

    base_times = None
    base_name = ""
    
    if has_image:
        base_times = np.array(image_times)
        base_name = "image"
        print(f'[PID:{pid} INFO] {bag_name}: Using IMAGE as synchronization base.')
    elif has_scan:
        base_times = np.array(scan_times)
        base_name = "scan"
        print(f'[PID:{pid} INFO] {bag_name}: Using SCAN as synchronization base.')

    cmd_data, cmd_times = np.array(cmd_data), np.array(cmd_times)
    
    if has_image and base_name != "image":
        image_data, image_times = np.array(image_data), np.array(image_times)
    
    if has_scan and base_name != "scan":
        scan_data, scan_times = np.array(scan_data), np.array(scan_times)

    if has_odom:
        odom_data, odom_times = np.array(odom_data), np.array(odom_times)

    synced_images, synced_scans = [], []
    synced_steers, synced_accelerations = [], []
    synced_odoms = []

    for i, base_time in enumerate(base_times):
        
        idx_cmd = np.argmin(np.abs(cmd_times - base_time))
        synced_steers.append(cmd_data[idx_cmd][0])
        synced_accelerations.append(cmd_data[idx_cmd][1])

        if has_image:
            if base_name == "image":
                synced_images.append(image_data[i])
            else:
                idx_image = np.argmin(np.abs(image_times - base_time))
                synced_images.append(image_data[idx_image])

        if has_scan:
            if base_name == "scan":
                synced_scans.append(scan_data[i])
            else:
                idx_scan = np.argmin(np.abs(scan_times - base_time))
                synced_scans.append(scan_data[idx_scan])

        if has_odom:
            idx_odom = np.argmin(np.abs(odom_times - base_time))
            synced_odoms.append(odom_data[idx_odom])

    
    if has_image:
        images_save_dir = out_dir / 'images'
        images_save_dir.mkdir(exist_ok=True)
        for i, image in enumerate(synced_images):
            image_filename = f"{i:06d}.png"
            image_save_path = images_save_dir / image_filename
            cv2.imwrite(str(image_save_path), image)

    np.save(out_dir / 'steers.npy', np.array(synced_steers))
    np.save(out_dir / 'accelerations.npy', np.array(synced_accelerations))
    
    if has_scan:
        np.save(out_dir / 'scans.npy', np.array(synced_scans))

    if has_odom:
        np.save(out_dir / 'odoms.npy', np.array(synced_odoms))

    print(f'[PID:{pid} SAVE] {bag_name}: {len(base_times)} samples (Base: {base_name}) saved to {out_dir}')

def main():
    """
    メイン関数。引数解析、バッグ検索、並列処理の起動を行う。
    """
    parser = argparse.ArgumentParser(description='Extract and synchronize image, scan, command, and odometry data from rosbags.')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--bags_dir', help='Path to directory containing rosbag folders (searches recursively)')
    group.add_argument('--seq_dirs', nargs='+', help='List of specific sequence directories to process (non-recursive)')

    parser.add_argument('--outdir', required=True, help='Output root directory')
    
    parser.add_argument('--image_topic', default='/realsense2_camera/color/image_raw', help='Image topic name (Optional, but image or scan is required)')
    parser.add_argument('--scan_topic', default='/scan', help='Scan (LaserScan) topic name (Optional, but image or scan is required)')
    parser.add_argument('--cmd_topic', default='/jetracer/cmd_drive', help='Command topic name (Required)')
    parser.add_argument('--odom_topic', default='/visual_slam/tracking/odometry', help='Odometry topic name (Optional)')
    
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers. (Default: CPU count - 1, max 8)')
    
    args = parser.parse_args()
    
    bag_dirs = []

    if args.bags_dir:
        print(f"[INFO] Mode: Recursive search in --bags_dir ({args.bags_dir})")
        bags_dir_path = Path(args.bags_dir).expanduser().resolve()
        
        for p in bags_dir_path.rglob("metadata.yaml"):
            if p.is_file():
                bag_dirs.append(p.parent)

        if not bag_dirs:
            print(f"[ERROR] No valid rosbag directories found in {bags_dir_path}.")
            if (bags_dir_path / "metadata.yaml").exists():
                print(f"[INFO] Treating {bags_dir_path} as a single bag directory.")
                bag_dirs = [bags_dir_path]
            else:
                return

    elif args.seq_dirs:
        print(f"[INFO] Mode: Direct processing of --seq_dirs ({len(args.seq_dirs)} items)")
        for seq_path_str in args.seq_dirs:
            seq_path = Path(seq_path_str).expanduser().resolve()
            
            if (seq_path / "metadata.yaml").is_file():
                bag_dirs.append(seq_path)
            elif seq_path.is_dir():
                print(f"[WARN] Skipping {seq_path.name} ({seq_path}): 'metadata.yaml' not found.")
            else:
                print(f"[WARN] Skipping {seq_path}: Directory not found.")

    if not bag_dirs:
        print("[ERROR] No valid rosbag directories to process.")
        return

    print(f"[INFO] Found {len(bag_dirs)} rosbag directories to process.")

    tasks = []
    for bag_path in sorted(bag_dirs):
        print(f"--- Queuing {bag_path.name} (from: {bag_path}) ---")
        task_args = (
            bag_path,
            args.outdir,
            args.image_topic,
            args.cmd_topic,
            args.odom_topic,
            args.scan_topic,
        )
        tasks.append(task_args)

    if args.workers:
        num_workers = args.workers
    else:
        cpu_count = os.cpu_count()
        if cpu_count:
            num_workers = min(max(1, cpu_count - 1), 8)
        else:
            num_workers = 4

    print(f"[INFO] Starting parallel processing with {num_workers} workers...")

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(extract_and_save_per_bag, tasks)

        print("[INFO] All processing finished.")

    except Exception as e:
        print(f"[ERROR] An error occurred during parallel processing: {e}")


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError as e:
        if "context has already been set" not in str(e):
            print(f"[WARN] Could not set start method 'spawn': {e}")
        pass

    main()