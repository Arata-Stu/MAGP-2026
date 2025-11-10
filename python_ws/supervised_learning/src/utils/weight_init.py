import torch.nn as nn
import torch.nn.init as init

def _init_weights(m):
    """
    モジュールに応じた重み初期化を適用する関数
    """
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        # 活性化関数がReLUなので、He初期化 (Kaiming Normal) を使用
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            # バイアスは0で初期化
            init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                # 入力-隠れ状態間の重み (Xavier Uniform)
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # 隠れ-隠れ状態間の重み (Orthogonal: 直交行列は再帰処理で勾配消失/爆発を防ぐのに効果的)
                init.orthogonal_(param.data)
            elif 'bias' in name:
                # バイアスは0で初期化
                param.data.fill_(0)
                # (Tips) forget gateのバイアスを1にすると、長期的な依存関係を学習しやすくなることがある
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)