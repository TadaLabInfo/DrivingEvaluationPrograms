import os
import copy
import cv2
import time
import numpy as np
import onnxruntime
from argparse import ArgumentParser
from typing import Tuple, Optional, List
from math import cos, inf, sin
from glob import glob
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


class GoldYOLOONNX(object):
    def __init__(
        self,
        model_path,
        providers,
        class_score_th: Optional[float] = 0.35,
    ):
        """GoldYOLOONNX

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for GoldYOLO

        class_score_th: Optional[float]
            Score threshold. Default: 0.35

        providers: Optional[List]
            Name of onnx execution providers
        """
        # スコアの閾値
        self.class_score_th = class_score_th

        # モデルの読み込み
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        # 使用するプロバイダーのリスト
        self.providers = self.onnx_session.get_providers()

        # モデルの入力シェイプ
        self.input_shapes = [input.shape for input in self.onnx_session.get_inputs()]
        # モデルの入力名
        self.input_names = [input.name for input in self.onnx_session.get_inputs()]
        # モデルの出力名
        self.output_names = [output.name for output in self.onnx_session.get_outputs()]

    def __call__(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """YOLOv7ONNX

        Parameters
        ----------
        frame: np.ndarray
            Entire frame

        Returns
        -------
        boxes: np.ndarray
            Predicted boxes: [N, y1, x1, y2, x2]

        scores: np.ndarray
            Predicted box scores: [N, score]
        """
        # 画像のコピーを作成
        temp_image = copy.deepcopy(frame)

        # 前処理
        resized_image = self.__preprocess(
            temp_image,
        )

        # 推論
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        boxes = self.onnx_session.run(
            self.output_names,
            {input_name: inferece_image for input_name in self.input_names},
        )[0]

        # 後処理
        result_boxes, result_scores = self.__postprocess(
            frame=temp_image,
            boxes=boxes,
        )

        return result_boxes, result_scores

    def __preprocess(
        self,
        frame: np.ndarray,
        swap: Optional[Tuple[int, int, int]] = (2, 0, 1),
    ) -> np.ndarray:
        """__preprocess

        Parameters
        ----------
        frame: np.ndarray
            Entire frame

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized frame.
        """
        # 正規化とBGRからRGBへの変換
        resized_image = cv2.resize(
            frame,
            (
                int(self.input_shapes[0][3]),
                int(self.input_shapes[0][2]),
            ),
        )
        resized_image = np.divide(resized_image, 255.0)
        resized_image = resized_image[..., ::-1]
        resized_image = resized_image.transpose(swap)
        resized_image = np.ascontiguousarray(
            resized_image,
            dtype=np.float32,
        )
        return resized_image

    def __postprocess(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """__postprocess

        Parameters
        ----------
        frame: np.ndarray
            Entire frame.

        boxes: np.ndarray
            float32[N, 7]

        Returns
        -------
        result_boxes: np.ndarray
            Predicted boxes: [N, y1, x1, y2, x2]

        result_scores: np.ndarray
            Predicted box confs: [N, score]
        """
        # 画像の高さと幅を取得
        image_height = frame.shape[0]
        image_width = frame.shape[1]

        """
        Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0

        batchno_classid_y1x1y2x2_score: float32[N,7]
        """
        result_boxes = []
        result_scores = []
        if len(boxes) > 0:
            # スコアを取得
            scores = boxes[:, 6:7]
            # スコアが閾値を超えるインデックスを取得
            keep_idxs = scores[:, 0] > self.class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                for box, score in zip(boxes_keep, scores_keep):
                    # ボックスの座標を計算
                    x_min = int(max(box[2], 0) * image_width / self.input_shapes[0][3])
                    y_min = int(max(box[3], 0) * image_height / self.input_shapes[0][2])
                    x_max = int(
                        min(box[4], self.input_shapes[0][3])
                        * image_width
                        / self.input_shapes[0][3]
                    )
                    y_max = int(
                        min(box[5], self.input_shapes[0][2])
                        * image_height
                        / self.input_shapes[0][2]
                    )

                    result_boxes.append([x_min, y_min, x_max, y_max])
                    result_scores.append(score)

        return np.asarray(result_boxes), np.asarray(result_scores)


class SixDRepNetCalculator:
    def __init__(
        self,
        use_cuda=False,
        yolo_model_path="gold_yolo_l_head_post_0277_0.5353_1x3x480x640.onnx",
        head_pose_model_path="sixdrepnet360_1x3x224x224_full.onnx",
        save_angle_video=False,
        save_dir="info",
    ):

        # providersの設定
        if use_cuda:
            self.providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            self.providers = [
                "CPUExecutionProvider",
            ]

        self.yolo_model = GoldYOLOONNX(
            model_path=yolo_model_path,
            providers=self.providers,
        )

        # セッションオプションの設定
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3  # INFO
        # 6DRepNetモデルのインスタンスを作成
        self.repnet_model = onnxruntime.InferenceSession(
            head_pose_model_path,
            sess_options=session_option,
            providers=self.providers,
        )

        self.save_angle_video = save_angle_video
        self.save_dir = save_dir

    def __call__(self, video_path, face_range=[0, 100], progress=None):
        """動画から顔の角度を取得する

        Raises:
            ValueError: 動画が開けない場合はエラーを出力
        """
        self.video_path = video_path
        # 動画の読み込み
        cap = cv2.VideoCapture(self.video_path)
            
        # 動画が開けない場合はエラーを出力
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        # 総フレーム数の取得（取得できない場合はNone）
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(total_frames) if total_frames > 0 else None

        # yawのリスト
        yaw_list = []
        # 重畳画像のリスト
        draw_img_list = []

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

        task = progress.add_task("6DRepNet処理中...", total=total_frames)
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
            yaw, draw_img = self.get_angle(frame)
            yaw_list.append(yaw)
            draw_img_list.append(draw_img)
            progress.update(task, advance=1)
        progress.remove_task(task)

        if self.save_angle_video:
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # video_pathからディレクトリ名を取得し、保存用の別名を作成
            save_dir = os.path.join(self.save_dir, "6drepnet")
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
                "6DRepNetの動画を保存中...", total=len(draw_img_list)
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

        return yaw_list

    def get_angle(self, frame):
        # 顔検出
        boxes, scores = self.yolo_model(frame)
        if len(boxes) > 0:
            x1y1x2y2cxcyidxes, normalized_image_rgbs = self.clip_face_image(
                frame, boxes
            )

            # 顔向き推論
            # yaw_list = []
            # yaw_pitch_rolls = []
            # for normalized_image_rgb in normalized_image_rgbs:
            #     normalized_image_rgb = normalized_image_rgb.reshape(1, 3, 224, 224)
            #     yaw_pitch_roll = self.repnet_model.run(
            #         None,
            #         {"input": np.asarray(normalized_image_rgb, dtype=np.float32)},
            #     )[0]
            #     print(yaw_pitch_roll[0])
            #     yaw_list.append(yaw_pitch_roll[0])
            #     yaw_pitch_rolls.append(yaw_pitch_roll)
            nomalized_image_rgb = normalized_image_rgbs[0].reshape(
                1, 3, 224, 224
            )  # 0人目の顔のみ
            yaw_pitch_roll = self.repnet_model.run(
                None,
                {"input": np.asarray(nomalized_image_rgb, dtype=np.float32)},
            )[0]
            yaw = -1 * yaw_pitch_roll[0][0]  # RepNetは右が正、左が負なので反転
            yaw_pitch_rolls = [yaw_pitch_roll]

            draw_img = None
            if self.save_angle_video:
                # 顔向き角度を描画
                draw_img = self.draw(frame, yaw_pitch_rolls, x1y1x2y2cxcyidxes)
            return yaw, draw_img
        else:
            return None, frame

    def draw(self, frame, yaw_pitch_rolls, x1y1x2y2cxcyidxes):
        for yaw_pitch_roll, x1y1x2y2cxcyidx in zip(yaw_pitch_rolls, x1y1x2y2cxcyidxes):
            yaw_pitch_roll = yaw_pitch_roll[0]
            yaw_deg = yaw_pitch_roll[0]  # ヨー角度
            pitch_deg = yaw_pitch_roll[1]  # ピッチ角度
            roll_deg = yaw_pitch_roll[2]  # ロール角度
            x1 = x1y1x2y2cxcyidx[0]  # 検出された顔の左上のx座標
            y1 = x1y1x2y2cxcyidx[1]  # 検出された顔の左上のy座標
            x2 = x1y1x2y2cxcyidx[2]  # 検出された顔の右下のx座標
            y2 = x1y1x2y2cxcyidx[3]  # 検出された顔の右下のy座標
            cx = x1y1x2y2cxcyidx[4]  # 顔の中心のx座標
            cy = x1y1x2y2cxcyidx[5]  # 顔の中心のy座標
            idx = x1y1x2y2cxcyidx[6]  # 顔のインデックス

            # 検出された顔の矩形を描画
            frame = cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )
            frame = cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                1,
            )

            # 顔のbboxの上にidxを描画
            frame = cv2.putText(
                frame,
                f"idx{idx}",
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 顔向きの軸を描画
            frame = self.draw_axis(
                frame,
                yaw_deg,
                pitch_deg,
                roll_deg,
                tdx=float(cx),
                tdy=float(cy),
                size=abs(x2 - x1) // 2,
            )
        return frame

    def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
        # 角度をラジアンに変換
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180
        if tdx is not None and tdy is not None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2
        # X軸（赤色）
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
        # Y軸（緑色）
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
        # Z軸（青色）
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy
        cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 4)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 4)
        return img

    def clip_face_image(self, frame, boxes):
        frame_width = frame.shape[1]  # フレームの幅
        frame_height = frame.shape[0]  # フレームの高さ

        x1y1x2y2cxcyidxes: List = []
        normalized_image_rgbs: List = []
        # 正規化のための平均と標準偏差
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        for idx, box in enumerate(boxes):
            x1: int = box[0]  # 検出された顔の左上のx座標
            y1: int = box[1]  # 検出された顔の左上のy座標
            x2: int = box[2]  # 検出された顔の右下のx座標
            y2: int = box[3]  # 検出された顔の右下のy座標

            # 中心座標と幅・高さを計算
            cx: int = (x1 + x2) // 2  # 顔の中心のx座標
            cy: int = (y1 + y2) // 2  # 顔の中心のy座標
            w: int = abs(x2 - x1)  # 顔の幅
            h: int = abs(y2 - y1)  # 顔の高さ
            ew: float = w * 1.2  # 顔の幅を1.2倍した拡張幅
            eh: float = h * 1.2  # 顔の高さを1.2倍した拡張高さ
            ex1 = int(cx - ew / 2)  # 拡張された顔領域の左上のx座標
            ex2 = int(cx + ew / 2)  # 拡張された顔領域の右下のx座標
            ey1 = int(cy - eh / 2)  # 拡張された顔領域の左上のy座標
            ey2 = int(cy + eh / 2)  # 拡張された顔領域の右下のy座標

            # スライス範囲を正規化(念の為)
            ey1, ey2 = min(ey1, ey2), max(ey1, ey2)
            ex1, ex2 = min(ex1, ex2), max(ex1, ex2)
            # 画像の範囲を超えないように調整
            ex1 = ex1 if ex1 >= 0 else 0
            ex2 = ex2 if ex2 <= frame_width else frame_width
            ey1 = ey1 if ey1 >= 0 else 0
            ey2 = ey2 if ey2 <= frame_height else frame_height

            # 顔画像を切り出し
            inference_image = frame.copy()
            head_image_bgr = inference_image[ey1:ey2, ex1:ex2, :]
            resized_image_bgr = cv2.resize(head_image_bgr, (256, 256))
            cropped_image_bgr = resized_image_bgr[16:240, 16:240, :]

            # 推論用の前処理
            cropped_image_rgb: np.ndarray = cropped_image_bgr[..., ::-1]
            normalized_image_rgb: np.ndarray = (cropped_image_rgb / 255.0 - mean) / std
            normalized_image_rgb = normalized_image_rgb.transpose(2, 0, 1)
            # normalized_image_rgb: np.ndarray = normalized_image_rgb[np.newaxis, ...]
            normalized_image_rgb: np.ndarray = normalized_image_rgb.astype(np.float32)

            x1y1x2y2cxcyidxes.append([x1, y1, x2, y2, cx, cy, idx])
            normalized_image_rgbs.append(normalized_image_rgb)

        return x1y1x2y2cxcyidxes, normalized_image_rgbs
