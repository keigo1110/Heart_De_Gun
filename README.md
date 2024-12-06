# Gun Gesture Heart Effect

このプロジェクトは、Webカメラ映像から手を検出し、特定の「銃を構える」ハンドジェスチャーを判定、さらに指の向きと発射動作をトリガーとして、画面上にハートエフェクトを表示するデモアプリケーションです。
銃を構えて狙いを定め（指を水平にして保持）、その状態から指を上向きに弾くことで「発射」判定となり、あらかじめ狙っていた方向にハートが拡大する演出が表示されます。

## 特徴

- **MediaPipe Handsを用いた手検出・ジェスチャー判定**: OpenCV + MediaPipeでリアルタイムな手ランドマークの検出を行います。
- **銃ポーズ検出**: 親指と人差し指を伸ばし、中指・薬指・小指を曲げることで「銃」の形状を認識します。
- **狙いを定めて発射**: 人差し指を水平に保持して狙いをつけ、その後指を上方向に動かすことで発射判定。
  発射後、狙った方向先にハートが拡大して出現します。
- **視覚的フィードバック**: 狙い中や発射判定、ハートエフェクトが画面上に表示され、ログ出力やデバッグメッセージで状況把握が可能です。

## 動作環境

- Python 3.7以上推奨
- OpenCV (opencv-python)
- MediaPipe
- NumPy

## インストール

```bash
pip install opencv-python mediapipe numpy
```

## 実行方法

```bash
python main.py
```

- カメラが起動し、手を映すと手のランドマークが表示されます。
- 親指と人差し指を伸ばし、それ以外の指を曲げる「銃ポーズ」をすると`Gun Pose`と表示されます。
- 人差し指を水平に構えると`Aiming...`と表示され、狙いが定まります。
- 狙いを定めた状態で人差し指を上方向に弾くように動かすと、`Shot Fired!`が表示され、狙った位置にハートが拡大表示されます。

## ファイル構成

- `bakyu-n.py`: メインスクリプト。
  カメラからフレーム取得、手の検出、銃ポーズ判定、狙いと発射動作のロジック、ハート描画処理が含まれます。

## カスタマイズ

- `forward_distance`: 狙った方向先に表示されるハートまでの距離を調整可能。
- `shot_frames`: ハートが表示され続けるフレーム数を調整可能。
- `move_threshold`: 指を上に弾く際の最小移動量しきい値を変更可能。

ソースコード中のクラス`GunGestureDetector`や`draw_heart`関数のパラメータを変更することで、動作を微調整できます。
