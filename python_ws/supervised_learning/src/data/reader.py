from pathlib import Path
import torch
import numpy as np
import cv2

class BaseSequenceReader:
    """
    各データセット構造に応じた抽象リーダ。
    実際には DummySequenceReader や DSECReader などが継承して実装される。
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.length = 0  

    def __len__(self):
        return self.length
    
    def load_frame(self, index: int):
        """
        各サブクラスで定義された load_xxx() 関数を keys_to_load に従って呼び出す。
        """
        raise NotImplementedError("Subclasses must implement load_frame()")
    

class RosbagSequenceReader(BaseSequenceReader):
    """
    前処理済みrosbagデータ（.png, .npy）を読み込むためのシーケンスリーダ。

    このクラスは、__init__ 時に .npy ファイルをメモリにロードします。
    画像ファイル（.png）は、呼び出し時（load_rgb）に都度読み込まれます。
    """

    def __init__(self, path: Path, keys_to_load: list):
        """
        Args:
            path (Path): シーケンスディレクトリのパス (例: .../dataset/seq_01)
            keys_to_load (list): 読み込むデータのキーリスト 
                                 (例: ["rgb", "scan", "steer"])
        """
        super().__init__(path)
        self.keys_to_load = keys_to_load
        
        # データファイルの存在フラグ
        self.has_image = (self.path / "images").is_dir()
        self.has_scan = (self.path / "scans.npy").exists()
        self.has_odom = (self.path / "odoms.npy").exists()
        self.has_steer = (self.path / "steers.npy").exists()
        self.has_acceleration = (self.path / "accelerations.npy").exists()

        if not self.has_steer or not self.has_acceleration:
            raise FileNotFoundError(
                f"Required files (steers.npy or accelerations.npy) not found in {self.path}"
            )

        # --- .npy ファイルをメモリにプリロード ---
        self.data_storage = {}
        if self.has_steer:
            self.data_storage["steer"] = np.load(self.path / "steers.npy")
        if self.has_acceleration:
            self.data_storage["accel"] = np.load(self.path / "accelerations.npy")
        if self.has_scan:
            self.data_storage["scan"] = np.load(self.path / "scans.npy")
        if self.has_odom:
            self.data_storage["odom"] = np.load(self.path / "odoms.npy")

        # --- データ長の決定 ---
        self.length = len(self.data_storage["steer"])

        # --- 画像ファイルのインデックス ---
        self.image_files = []
        if self.has_image:
            # ファイル名をソートしてインデックスと一致させる
            self.image_files = sorted((self.path / "images").glob("*.png"))
            if len(self.image_files) != self.length:
                print(f"[WARN] {self.path.name}: Length mismatch! "
                      f"steers.npy ({self.length}) vs images ({len(self.image_files)})")
                
        self._validate_keys()


    def _validate_keys(self):
        """keys_to_load が実際にロード可能か検証する"""
        for key in self.keys_to_load:
            if not hasattr(self, f"load_{key}"):
                raise AttributeError(f"Requested key '{key}' has no corresponding load_{key}() method.")
            
            if key == "rgb" and not self.has_image:
                raise FileNotFoundError(f"Requested key 'rgb' but 'images' directory not found in {self.path}")
            if key == "scan" and not self.has_scan:
                raise FileNotFoundError(f"Requested key 'scan' but 'scans.npy' not found in {self.path}")
            if key == "odom" and not self.has_odom:
                raise FileNotFoundError(f"Requested key 'odom' but 'odoms.npy' not found in {self.path}")


    def load_rgb(self, index: int):
        """
        画像を読み込み、正規化して (C, H, W) テンソルで返す。
        """
        if index >= len(self.image_files):
             raise IndexError(f"Index {index} out of bounds for images (len: {len(self.image_files)})")
        
        img_path = self.image_files[index]
        # OpenCV は BGR 形式で読み込む
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR) 
        if img_bgr is None:
            raise IOError(f"Failed to load image: {img_path}")
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # [0, 255] (uint8) -> [0.0, 1.0] (float32)
        img_float = np.asarray(img_rgb, dtype=np.float32) / 255.0
        
        # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img_float).permute(2, 0, 1)
        return tensor

    def load_scan(self, index: int):
        """プリロードしたスキャンデータからスライスを取得"""
        scan_data = self.data_storage["scan"][index]
        scan_data = scan_data / 30.0
        return torch.from_numpy(scan_data.astype(np.float32))

    def load_steer(self, index: int):
        """プリロードしたステアリングデータから値を取得"""
        steer_data = self.data_storage["steer"][index]
        # スカラー値を 0-dim テンソルに変換
        return torch.tensor(steer_data, dtype=torch.float32)

    def load_accel(self, index: int):
        """プリロードした加速度データから値を取得"""
        accel_data = self.data_storage["accel"][index]
        # スカラー値を 0-dim テンソルに変換
        return torch.tensor(accel_data, dtype=torch.float32)

    def load_odom(self, index: int):
        """プリロードしたオドメトリデータからスライスを取得"""
        odom_data = self.data_storage["odom"][index]
        # (7,) 形式の numpy 配列をテンソルに変換
        return torch.from_numpy(odom_data.astype(np.float32))

    def load_frame(self, index: int):
        """
        keys_to_load に指定されたセンサー群をまとめてロードし、辞書で返す。
        
        Args:
            index (int): フレームインデックス
            
        Returns:
            dict: データの辞書 (例: {"rgb": Tensor, "scan": Tensor, ...})
        """
        if index < 0:
            index = self.length + index  
            
        if index >= self.length or index < 0:
            raise IndexError(f"Index {index} out of range (Sequence length: {self.length})")

        frame_data = {}
        for key in self.keys_to_load:
            func = getattr(self, f"load_{key}")
            frame_data[key] = func(index)
        
        return frame_data