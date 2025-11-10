#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math, os, time

class ScanViewer(Node):
    def __init__(self):
        super().__init__('scan_cui_viewer')
        self.create_subscription(LaserScan, '/scan', self.callback, 10)
        self.term_size = os.get_terminal_size()
        self.cx = self.term_size.columns // 2
        self.cy = self.term_size.lines // 2
        self.last_msg_time = time.time()
        self.msg_timeout = 3.0 # 3秒メッセージが来なかったらクリア

    def callback(self, msg):
        self.last_msg_time = time.time()
        try:
            self.term_size = os.get_terminal_size()
            self.cx = self.term_size.columns // 2
            self.cy = self.term_size.lines // 2

            print("\033[2J\033[H", end='') # 画面クリア
            print("=== LaserScan BEV Viewer (/scan) ===\n")
            
            max_r = msg.range_max if msg.range_max > 0 else 10.0
            scale_x = (self.cx - 1) / max_r
            scale_y = (self.cy - 2) / max_r # ヘッダー分を考慮
            scale = min(scale_x, scale_y)
            
            points_drawn = 0
            buffer_h = self.term_size.lines - 2 # ヘッダーとフッター
            buffer_w = self.term_size.columns
            if buffer_h <= 0 or buffer_w <= 0:
                return # ターミナルが小さすぎる
                
            header_offset = 2 
            output = []
            
            step = 8 
            for i in range(0, len(msg.ranges), step):
                r = msg.ranges[i]
                if math.isinf(r) or math.isnan(r) or r <= msg.range_min:
                    continue
                    
                angle = msg.angle_min + i * msg.angle_increment
                x = int(self.cx + r * scale * math.cos(angle))
                y = int(self.cy - r * scale * math.sin(angle)) + header_offset
                
                if (header_offset <= y < self.term_size.lines) and (0 <= x < self.term_size.columns):
                    output.append(f"\033[{y+1};{x+1}H*")
                    points_drawn += 1

            footer_line = self.term_size.lines
            output.append(f"\033[{footer_line};0H")
            output.append(f"Range: {msg.range_min:.1f}-{max_r:.1f}m | Points: {points_drawn}/{len(msg.ranges)//step}")
            
            print("".join(output), end='', flush=True)

        except Exception as e:
            print(f"\033[2J\033[HAn error occurred: {e}", end='')
            time.sleep(1)

    def check_timeout(self):
        # メッセージがタイムアウトしたら画面をクリア
        if time.time() - self.last_msg_time > self.msg_timeout:
            try:
                self.term_size = os.get_terminal_size()
                print("\033[2J\033[H", end='') 
                print("=== LaserScan BEV Viewer (/scan) ===\n")
                print("\n\n... データを待機中 ...")
                print(f"\033[{self.term_size.lines};0HTimeout: No /scan msg received.", end='', flush=True)
            except Exception:
                pass 

def main():
    rclpy.init()
    node = ScanViewer()
    
    timer = node.create_timer(1.0, node.check_timeout)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\033[2J\033[HScan viewer stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()