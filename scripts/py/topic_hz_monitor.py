#!/usr/bin/env python3
import rclpy, os, time, subprocess, sys, threading, math
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.utilities import get_message

# {topic_name: (last_time, count, hz, msg_type)}
topic_stats = {}
lock = threading.Lock()

class GenericSubscriber(Node):
    def __init__(self, topics_to_monitor):
        super().__init__('topic_hz_monitor')
        self.topics = topics_to_monitor
        self.subs = []
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        for topic_name, msg_type in self.topics:
            with lock:
                topic_stats[topic_name] = [time.time(), 0, 0.0, msg_type]
            try:
                msg_class = get_message(msg_type)
                callback_lambda = lambda msg, name=topic_name: self.topic_callback(msg, name)
                self.create_subscription(
                    msg_class, topic_name, callback_lambda, qos_profile
                )
            except Exception as e:
                with lock:
                    # サブスクリプション失敗
                    topic_stats[topic_name] = [time.time(), 0, -1.0, msg_type] 

    def topic_callback(self, msg, topic_name):
        with lock:
            now = time.time()
            if topic_name in topic_stats:
                stats = topic_stats[topic_name]
                stats[0] = now # last_time
                stats[1] += 1 # count

def display_loop(update_interval=1.0):
    """ HZを計算してCUI表示するループ """
    # {topic_name: last_count}
    last_counts = {}
    
    try:
        while True:
            time.sleep(update_interval)
            print("\033[2J\033[H", end='') # 画面クリア
            print("=== Topic HZ Monitor ===\n")
            print(f"{'Topic Name':<40} {'Msg Type':<30} {'HZ'}")
            print(f"{'-'*40:<40} {'-'*30:<30} {'-'*10}")

            with lock:
                if not topic_stats:
                    print("トピックが選択されていません。")
                    continue

                for name, stats in topic_stats.items():
                    last_time, current_count, hz, msg_type = stats
                    
                    if name not in last_counts:
                        last_counts[name] = (current_count, last_time)
                        
                    last_c, last_t = last_counts[name]
                    
                    now = time.time()
                    time_diff = now - last_t
                    count_diff = current_count - last_c
                    
                    current_hz = 0.0
                    if time_diff > 0:
                        current_hz = count_diff / time_diff
                    
                    # 統計情報を更新
                    stats[2] = current_hz
                    last_counts[name] = (current_count, now)
                    
                    # 3秒以上更新がなければHZを0とみなす
                    if now - last_time > 3.0:
                         stats[2] = 0.0

                    hz_val = stats[2]
                    hz_str = ""
                    if hz_val < 0:
                        hz_str = "ERROR"
                    else:
                        hz_str = f"{hz_val:5.1f}"

                    print(f"{name:<40} {msg_type:<30} {hz_str:>10}")

            print("\n(tmuxペインを閉じるには Ctrl+B -> X)")
            
    except KeyboardInterrupt:
        pass

def main():
    # 呼び出し元 (dashboard.sh) が dialog でトピックを選択し、
    # 引数として (name,msg_type) の形式で渡す
    
    if len(sys.argv) <= 1:
        print("エラー: 監視対象トピックが指定されていません。", file=sys.stderr)
        print("このスクリプトは dashboard.sh から呼び出されることを想定しています。", file=sys.stderr)
        sys.exit(1)
        
    selected_topics = []
    for arg in sys.argv[1:]:
        try:
            name, msg_type = arg.split(',')
            selected_topics.append((name, msg_type))
        except Exception:
            print(f"引数フォーマットエラー: {arg}", file=sys.stderr)
            
    if not selected_topics:
        print("有効なトピックが指定されませんでした。", file=sys.stderr)
        sys.exit(1)

    rclpy.init()
    node = GenericSubscriber(selected_topics)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,))
    spin_thread.daemon = True
    spin_thread.start()
    
    display_loop() # メインスレッドで表示
    
    # 終了処理
    rclpy.shutdown()
    spin_thread.join()
    print("\nTopic HZ monitor stopped.")

if __name__ == '__main__':
    main()