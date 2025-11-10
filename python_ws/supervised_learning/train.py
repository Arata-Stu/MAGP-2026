import torch
from omegaconf import OmegaConf, DictConfig
import hydra
import os
from torch.utils.tensorboard import SummaryWriter

from src.data.reader import RosbagSequenceReader
from src.data.stream_dataset import build_stream_dataset
from src.data.mixed_dataloader import MixedDataLoader, concat_mixed_batches
from src.data.types import Mode, ModelMode
from src.model.model import load_model
from src.utils.rnn_state import RNNStates

class Trainer:
    def __init__(self, cfg: DictConfig):
        """
        Trainerクラスの初期化。
        設定の読み込み、デバイス、ロガー、データローダー、モデル等を準備します。
        """
        self.cfg = cfg
        print("--- Configuration ---")
        print(OmegaConf.to_yaml(cfg))
        print("---------------------")

        # --- 1. デバイスとロガーの設定 ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.log_dir = hydra.utils.to_absolute_path(cfg.log_dir)
        self.ckpt_dir = hydra.utils.to_absolute_path(cfg.ckpt_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"Logs will be saved to: {self.log_dir}")
        print(f"Checkpoints will be saved to: {self.ckpt_dir}")

        # --- 2. データローダーの準備 ---
        self.mixed_train_loader = MixedDataLoader(
            dataset_config=cfg.data,
            reader_cls=RosbagSequenceReader,
            dataset_mode=Mode.TRAIN
        )
        self.stream_val_loader = build_stream_dataset(
            dataset_config=cfg.data,
            reader_cls=RosbagSequenceReader,
            dataset_mode=Mode.VALIDATION,
        )

        # --- 3. モデル、損失関数、オプティマイザ ---
        self.model = load_model(cfg.model).to(self.device)
        self.model_mode = self.model.model_type
        self.is_rnn = self.model.is_rnn

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.training.learning_rate)

        # --- 4. RNN状態管理 ---
        self.mode_2_rnn_states = {
            Mode.TRAIN: RNNStates(),
            Mode.VALIDATION: RNNStates(),
        }

        # --- 5. 訓練状態とパラメータ ---
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.max_steps = cfg.training.max_steps
        self.val_interval = cfg.training.val_interval_steps
        self.log_interval = cfg.training.log_interval

    def _train_step(self, mixed_data):
        """
        単一の訓練ステップを実行します。
        """
        # --- データ準備 ---
        random_data = mixed_data["random"]
        stream_data = mixed_data["stream"]
        mixed_batch = concat_mixed_batches(random_data, stream_data)
        data = mixed_batch["data"]
        worker_id = mixed_batch["worker_id"]

        image_seq = data["rgb"].to(self.device) if self.model_mode == ModelMode.IMAGE or self.model_mode == ModelMode.FUSION else None
        scan_seq = data["scan"].to(self.device) if self.model_mode == ModelMode.LIDAR or self.model_mode == ModelMode.FUSION else None
        steer_seq = data["steer"]
        accel_seq = data["accel"]
        is_first_sample = data["is_first_sample"]

        target_seq = torch.cat([steer_seq, accel_seq], dim=-1).to(self.device)

        # --- RNN状態リセットとフォワードパス ---
        if self.is_rnn:
            self.mode_2_rnn_states[Mode.TRAIN].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
            prev_states = self.mode_2_rnn_states[Mode.TRAIN].get_states(worker_id=worker_id)

        if self.model_mode == ModelMode.IMAGE:
            if self.is_rnn:
                output, states = self.model(image_seq, prev_states)
                self.mode_2_rnn_states[Mode.TRAIN].save_states_and_detach(worker_id=worker_id, states=states)
            else:
                output = self.model(image_seq)
        elif self.model_mode == ModelMode.LIDAR:
            if self.is_rnn:
                output, states = self.model(scan_seq, prev_states)
                self.mode_2_rnn_states[Mode.TRAIN].save_states_and_detach(worker_id=worker_id, states=states)
            else:
                output = self.model(scan_seq)
        elif self.model_mode == ModelMode.FUSION:
            if self.is_rnn:
                output, states = self.model(image_seq, scan_seq, prev_states)
                self.mode_2_rnn_states[Mode.TRAIN].save_states_and_detach(worker_id=worker_id, states=states)
            else:
                output = self.model(image_seq, scan_seq)

        # --- 損失計算 ---
        if output.dim() == 3 and output.shape[1] > 1:
            target = target_seq
        elif output.dim() == 2:
            target = target_seq[:, -1, :]
        else:
            raise ValueError(f"Unsupported output shape: {output.shape}")

        loss = self.criterion(output, target)

        # --- 逆伝播と最適化 ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _run_validation(self):
        """
        検証ループを実行し、モデルを評価・保存します。
        """
        print(f"\n--- Running validation at step {self.global_step} ---")
        self.model.eval() 
        running_val_loss = 0.0

        with torch.no_grad():
            for val_batch in self.stream_val_loader:
                
                val_data = val_batch["data"]
                val_worker_id = val_batch["worker_id"]

                image_seq = val_data["rgb"].to(self.device) if self.model_mode == ModelMode.IMAGE or self.model_mode == ModelMode.FUSION else None
                scan_seq = val_data["scan"].to(self.device) if self.model_mode == ModelMode.LIDAR or self.model_mode == ModelMode.FUSION else None
                steer_seq = val_data["steer"]
                accel_seq = val_data["accel"]
                is_first_sample = val_data["is_first_sample"]

                target_seq = torch.cat([steer_seq, accel_seq], dim=-1).to(self.device)

                if self.is_rnn:
                    self.mode_2_rnn_states[Mode.VALIDATION].reset(worker_id=val_worker_id, indices_or_bool_tensor=is_first_sample)
                    prev_states = self.mode_2_rnn_states[Mode.VALIDATION].get_states(worker_id=val_worker_id)
                
                # --- モデルフォワード (検証) ---
                if self.model_mode == ModelMode.IMAGE:
                    if self.is_rnn:
                        output, states = self.model(image_seq, prev_states)
                        self.mode_2_rnn_states[Mode.VALIDATION].save_states_and_detach(worker_id=val_worker_id, states=states)
                    else:
                        output = self.model(image_seq)
                elif self.model_mode == ModelMode.LIDAR:
                    if self.is_rnn:
                        output, states = self.model(scan_seq, prev_states)
                        self.mode_2_rnn_states[Mode.VALIDATION].save_states_and_detach(worker_id=val_worker_id, states=states)
                    else:
                        output = self.model(scan_seq)
                elif self.model_mode == ModelMode.FUSION:
                    if self.is_rnn:
                        output, states = self.model(image_seq, scan_seq, prev_states)
                        self.mode_2_rnn_states[Mode.VALIDATION].save_states_and_detach(worker_id=val_worker_id, states=states)
                    else:
                        output = self.model(image_seq, scan_seq)
                
                # --- 損失計算 (検証) ---
                if output.dim() == 3 and output.shape[1] > 1:
                    target = target_seq
                elif output.dim() == 2:
                    target = target_seq[:, -1, :]
                else:
                    raise ValueError(f"Unsupported output shape: {output.shape}")

                loss = self.criterion(output, target)
                running_val_loss += loss.item()

        # --- 検証結果の集計・ログ・保存 ---
        if len(self.stream_val_loader) > 0:
            step_val_loss = running_val_loss / len(self.stream_val_loader)
        else:
            step_val_loss = 0.0
            print("Warning: Validation loader is empty.")

        self.writer.add_scalar('Loss/validation', step_val_loss, self.global_step)
        print(f"Validation Loss at Step {self.global_step}: {step_val_loss:.4f}")

        # 1. 常に最新の重みを 'last.pth' として保存
        last_ckpt_path = os.path.join(self.ckpt_dir, "last.pth")
        torch.save(self.model.state_dict(), last_ckpt_path)

        # 2. 検証ロスが改善した場合のみ 'best.pth' として保存
        if step_val_loss < self.best_val_loss and len(self.stream_val_loader) > 0:
            self.best_val_loss = step_val_loss
            best_ckpt_path = os.path.join(self.ckpt_dir, "best.pth")
            torch.save(self.model.state_dict(), best_ckpt_path)
            print(f"New best model saved to {best_ckpt_path} (Val Loss: {self.best_val_loss:.4f})")
        
        print("--- Validation complete ---")
        self.model.train() 

    def run_training(self):
        """
        メインの訓練ループを実行します。
        """
        print("Starting training...")
        
        training_complete = False
        
        # エポックはデータローダーを無限ループさせるために使用
        for epoch in range(self.cfg.training.epochs):
            
            if training_complete:
                break

            self.model.train()
            
            for i, mixed_data in enumerate(self.mixed_train_loader):
                
                # 訓練ステップ実行
                loss = self._train_step(mixed_data)
                self.global_step += 1

                # ステップベースのログ出力
                if self.global_step % self.log_interval == 0:
                    self.writer.add_scalar('Loss/train_step', loss, self.global_step)
                    print(f"Epoch [{epoch+1}], Step [{self.global_step}], Train Loss: {loss:.4f}")

                # ステップベースの検証
                if self.global_step % self.val_interval == 0:
                    self._run_validation()

                # 終了条件のチェック
                if self.global_step >= self.max_steps:
                    print(f"Reached max_steps ({self.max_steps}). Finishing training.")
                    training_complete = True
                    break
        
        # --- 最終検証と保存 ---
        if self.global_step % self.val_interval != 0: 
            print("Running final validation...")
            self._run_validation()

        # --- 終了処理 ---
        self.writer.close()
        print("---------------------")
        print("Training complete.")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Find logs at: {self.log_dir}")
        print(f"Find checkpoints at: {self.ckpt_dir}")

# --- Hydraのエントリポイント ---
@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.run_training()

if __name__ == "__main__":
    main()