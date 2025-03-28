import cv2
import numpy as np
import insightface
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)
import sys
import os
import warnings

warnings.filterwarnings("ignore")


# 標準出力を一時的に無効化
class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")  # 標準出力を/dev/nullにリダイレクト

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self._original_stdout  # 標準出力を元に戻す


class InsightFaceCalculator:
    def __init__(self, use_cuda=False, save_angle_video=False, save_dir="info"):
        """InsightFaceを利用して動画から顔の角度を取得する

        Args:
            use_cuda (bool, optional): GPUを利用するかどうか. Defaults to False.
        """
        with SuppressPrint():
            self.app = insightface.app.FaceAnalysis()  # 顔認識オブジェクト生成
            if use_cuda:
                self.app.prepare(ctx_id=0)  # GPUを利用する場合
            else:
                self.app.prepare(ctx_id=-1)  # CPUを利用する場合

        self.save_angle_video = save_angle_video
        self.save_dir = save_dir

    def __call__(self, video_path, face_range=[0, 100], progress=None):
        """動画から顔の角度を取得する

        Args:
            video_path (str): 動画のパス
            face_range (list, optional): 顔のフレーム範囲で、左30%のみを取得するならば[0, 30]を指定. Defaults to [0, 100].

        Raises:
            ValueError: 動画が開けない場合はエラーを出力
        """
        # 動画の読み込み
        cap = cv2.VideoCapture(video_path)

        # 動画が開けない場合はエラーを出力
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # 全フレーム数を取得（進捗表示のため）
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # yawのリスト
        yaw_list = []
        # 重畳画像のリスト
        draw_img_list = []
        # det_scoreのリスト
        det_score_list = []

        if progress is None:
            progress = Progress(
                SpinnerColumn(),
                "[progress.description]{task.description}",
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[frame]}"),
            )

        task = progress.add_task("InsightFace処理中...", total=total_frames)
        # 動画のフレーム数分ループ
        while True:
            # フレームを1つ読み込む
            ret, frame = cap.read()
            if not ret:
                # 動画の終わりに到達した場合
                break

            # face_rangeで指定した横方向の範囲と動画の下半分のみを取得
            frame = frame[
                int(frame.shape[0] / 2):,  # 下半分を切り取る
                int(frame.shape[1] * face_range[0] / 100) : int(
                    frame.shape[1] * face_range[1] / 100
                ),
            ]

            # 顔向き角度と重畳画像の検出
            yaw, draw_img, det_score = self.get_angle(frame)
            yaw_list.append(yaw)
            draw_img_list.append(draw_img)
            det_score_list.append(det_score)

            progress.update(task, advance=1)
        progress.remove_task(task)

        if self.save_angle_video:
            h, w = draw_img_list[0].shape[:2]
            # video_pathからディレクトリ名を取得し、保存用の別名を作成
            save_dir = os.path.join(self.save_dir, "insight")
            os.makedirs(save_dir, exist_ok=True)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_path = os.path.join(save_dir, video_name + ".mp4")
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                cap.get(cv2.CAP_PROP_FPS),
                (w, h),
            )
            
            task = progress.add_task(
                "InsightFaceの動画を保存中...", total=len(draw_img_list)
            )
            # 重畳画像を動画に書き込み
            for img, yaw in zip(draw_img_list, yaw_list):
                # 画像の右上に角度を表示
                if yaw is not None:
                    # 右上の座標を取得
                    h, w = img.shape[:2]
                    # 右上の座標に角度を描画
                    cv2.putText(
                        img=img,
                        text=f"yaw: {yaw:.1f}",
                        org=(int(w * 0.4), int(h * 0.1)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=(0, 255, 0),
                        thickness=3,
                        lineType=cv2.LINE_AA,
                    )
                writer.write(img)
                progress.update(task, advance=1)
            progress.remove_task(task)
            writer.release()

        # リソースを解放
        cap.release()
        return yaw_list, det_score_list

    def get_angle(self, frame):
        # 顔を取得
        faces = self.app.get(frame)

        if len(faces) == 0:
            return None, None, 0

        draw_img = None
        if self.save_angle_video:
            # 描画
            draw_img = self.app.draw_on(frame, faces)

        yaw = faces[0].pose[1]  # 0人目の顔向き角度(yawのみ)を取得
        det_score = faces[0].det_score  # 信頼度スコアを追加

        return yaw, draw_img, det_score
