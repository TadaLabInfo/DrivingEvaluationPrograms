import subprocess
import re
import pandas as pd
import os
import numpy as np
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

from glob import glob
from .face_angle_handler_parallel import FaceAngleHandler

import logging
import coloredlogs

coloredlogs.install(level="DEBUG")

class InfoExtractor:
    """GPSデータと顔向き角度(以下、コメントでは情報と呼称)を動画ファイルから抽出するクラス(GPSはexiftoolを使用)"""

    dir_path = None  # 処理するディレクトリのパス
    video_paths = None  # ディレクトリ内の動画ファイルのパス
    info_save_dir = None  # 情報を保存先ディレクトリ
    current_video_path = None  # 処理する動画ファイルのパス

    def __init__(
        self,
        dir_path,
        info_save_dir=None,
        use_cuda=False,
        parallel=False,
        save_angle_video=False,
        save_angle_csv=False,
        sampling_rate=15,
        face_range=[0, 100],
        replace_smooth=True,
        smooth=True,
    ):
        """
        InfoExtractorクラスの初期化
        Args:
            dir_path (str): 処理するディレクトリのパス
            info_save_dir (str): 情報保存先ディレクトリ(Noneの場合はディレクトリ名 + "_info")
        """
        self.dir_path = dir_path
        self.video_paths = sorted(glob(f"{dir_path}/*.AVI"))
        if info_save_dir is None:
            self.info_save_dir = dir_path + "_info"
        else:
            self.info_save_dir = info_save_dir
        os.makedirs(self.info_save_dir, exist_ok=True)

        self.face_angle_handler = FaceAngleHandler(
            use_cuda=use_cuda,
            parallel=parallel,
            save_angle_video=save_angle_video,
            save_angle_csv=save_angle_csv,
            save_dir=self.info_save_dir,
            face_range=face_range,
            replace_smooth=replace_smooth,
            smooth=smooth,
        )
        
        self.sampling_rate = sampling_rate

        self.logger = logging.getLogger("InfoExtractor")

    def __call__(self):
        """情報を抽出する"""
        all_info_data_raw = []
        all_info_data_removed_duplicates = []
        all_info_data_interpolate = []
        all_info_data_interpolate_fixed = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            # 情報抽出の進捗バー
            task = progress.add_task(
                "動画からGPSデータと顔向き角度の抽出中...", total=len(self.video_paths)
            )

            insight_face_angles = []
            insight_face_det_scores = []
            sixdrepnet_angles = []
            # 各動画ファイルについて情報を抽出
            for video_path in self.video_paths:
                # 動画ファイルのパスを保存
                self.current_video_path = video_path
                self.logger.debug(f"処理中の動画ファイルのパス: {video_path}")
                # 顔向きデータの抽出と保存(中央値計算のため)
                insight_face_angle, insight_face_det_score, sixdrepnet_angle = (
                    self.face_angle_handler(video_path, progress=progress)
                )
                insight_face_angles.append(insight_face_angle)
                insight_face_det_scores.append(insight_face_det_score)
                sixdrepnet_angles.append(sixdrepnet_angle)
                # 生GPSデータの抽出
                info_df_raw = self.extract_raw_gps_data()
                # 生GPSデータをリストへ保存
                all_info_data_raw.append(info_df_raw)
                # 重複削除GPSデータの抽出
                info_df_removed_duplicates = self.remove_duplicates(info_df_raw)
                # 重複削除GPSデータをリストへ保存
                all_info_data_removed_duplicates.append(info_df_removed_duplicates)
                # 補間GPSデータの抽出
                info_df_interpolate = self.interpolate_gps_data(
                    info_df_removed_duplicates, self.sampling_rate
                )
                # 補間GPSデータをリストへ保存
                all_info_data_interpolate.append(info_df_interpolate)
                # 進捗バーを進める
                progress.advance(task)

            self.logger.info("顔向き角度の中央値を計算中...")
            # 顔向き角度の中央値を計算
            insight_median = self.face_angle_handler.calc_yaw_median(
                insight_face_angles
            )
            sixdrepnet_median = self.face_angle_handler.calc_yaw_median(
                sixdrepnet_angles
            )

            self.logger.info(
                f"顔向き角度の中央値: InsightFace: {insight_median}, SixDRepNet: {sixdrepnet_median}"
            )

            self.logger.info("顔向き角度の修正中...")
            # 顔向き角度の修正
            for (
                insight_face_angle,
                insight_face_det_score,
                sixdrepnet_angle,
                gps_data_interpolate,
                video_path,
            ) in zip(
                insight_face_angles,  # InsightFaceの顔向き角度群
                insight_face_det_scores,  # InsightFaceの顔向き角度の信頼度群
                sixdrepnet_angles,  # SixDRepNetの顔向き角度群
                all_info_data_interpolate,  # 角度を追記するGPSデータ群
                self.video_paths,  # 動画ファイルのパス
            ):
                fixed_yaw_df = self.face_angle_handler.fix(
                    insight_yaw=insight_face_angle,  # InsightFaceの顔向き角度
                    insight_det=insight_face_det_score,  # InsightFaceの顔向き角度の信頼度
                    repnet_yaw=sixdrepnet_angle,  # SixDRepNetの顔向き角度
                    insight_median=insight_median,  # InsightFaceの顔向き角度の中央値
                    repnet_median=sixdrepnet_median,  # SixDRepNetの顔向き角度の中央値
                    gps_df_interpolate=gps_data_interpolate,  # 角度を追記するGPSデータ
                    save_path=video_path,  # 顔向き角度のグラフを保存するパス
                )
                all_info_data_interpolate_fixed.append(fixed_yaw_df)

            progress.update(
                task, description="[green]動画からGPSデータと顔向き角度の抽出中...完了"
            )

        columns = ["date_time", "latitude", "longitude", "speed", "video_path"]
        # 生GPSデータの全結合
        combined_gps_df_raw = pd.concat(all_info_data_raw, ignore_index=True)
        # 生GPSデータのパス生成
        combined_save_path = os.path.join(
            self.info_save_dir, "combined_raw_gps_data.csv"
        )
        # 生GPSデータの保存
        combined_gps_df_raw.to_csv(
            combined_save_path,
            columns=columns,
            index=False,
        )

        # 重複削除GPSデータの全結合
        combined_gps_df_removed_duplicates = pd.concat(
            all_info_data_removed_duplicates, ignore_index=True
        )
        # 重複削除GPSデータのパス生成
        combined_save_path = os.path.join(
            self.info_save_dir, "combined_removed_duplicates_gps_data.csv"
        )
        # 重複削除GPSデータの保存
        combined_gps_df_removed_duplicates.to_csv(
            combined_save_path,
            columns=columns,
            index=False,
        )

        columns = [
            "date_time",
            "latitude",
            "longitude",
            "speed",
            "video_path",
            "face_angle",
            "6dreplaced",
        ]
        # 補完 & 顔向き追加GPSデータの全結合
        combined_gps_df_fixed = pd.concat(
            all_info_data_interpolate_fixed, ignore_index=True
        )
        # 補完 & 顔向き追加GPSデータのパス生成
        combined_save_path = os.path.join(
            self.info_save_dir, "combined_interpolated_fixed_gps_data.csv"
        )
        # 補完 & 顔向き追加GPSデータの保存
        combined_gps_df_fixed.to_csv(
            combined_save_path,
            columns=columns,
            index=False,
        )

        return combined_gps_df_fixed

    def extract_raw_gps_data(self, command=["exiftool", "-ee", "-G3"]):
        """exiftoolコマンドを実行してfile_pathの生のGPSデータを抽出する

        Args:
            command (list): exiftoolコマンドのリスト

        Returns:
            pandas.DataFrame: 生のGPSデータを格納したDataFrame
                {
                    "date_time": [GPS Date/Timeのリスト],
                    "latitude": [GPS Latitudeのリスト],
                    "longitude": [GPS Longitudeのリスト],
                    "speed": [GPS Speedのリスト],
                    "video_path": [動画ファイルのパス]
                }
        """
        command = command + [self.current_video_path]
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        output = result.stdout
        gps_date_time = []
        gps_latitude = []
        gps_longitude = []
        gps_speed = []

        date_time_pattern = re.compile(r"GPS Date/Time\s+:\s+(.+)")
        latitude_pattern = re.compile(r"GPS Latitude\s+:\s+(.+)")
        longitude_pattern = re.compile(r"GPS Longitude\s+:\s+(.+)")
        speed_pattern = re.compile(r"GPS Speed\s+:\s+(\d+)")

        for line in output.splitlines():
            date_time_match = date_time_pattern.search(line)
            latitude_match = latitude_pattern.search(line)
            longitude_match = longitude_pattern.search(line)
            speed_match = speed_pattern.search(line)

            if date_time_match:
                gps_date_time.append(date_time_match.group(1))
            if latitude_match:
                gps_latitude.append(latitude_match.group(1))
            if longitude_match:
                gps_longitude.append(longitude_match.group(1))
            if speed_match:
                gps_speed.append(speed_match.group(1))

        gps_df = pd.DataFrame(
            {
                "date_time": gps_date_time,
                "latitude": gps_latitude,
                "longitude": gps_longitude,
                "speed": gps_speed,
                "video_path": self.current_video_path,
            }
        )

        save_dir_path = os.path.join(self.info_save_dir, "raw")
        os.makedirs(save_dir_path, exist_ok=True)
        save_file_path = os.path.splitext(os.path.basename(self.current_video_path))[0]
        save_file_path = os.path.join(save_dir_path, save_file_path + ".csv")
        gps_df.to_csv(
            save_file_path,
            columns=["date_time", "latitude", "longitude", "speed", "video_path"],
            index=False,
        )

        return gps_df

    def remove_duplicates(self, gps_df):
        """GPSデータの重複があれば削除する

        Args:
            gps_df (pandas.DataFrame): GPSデータを格納したDataFrame
                { "date_time": [GPS Date/Timeのリスト],
                "latitude":  [GPS Latitudeのリスト],
                "longitude": [GPS Longitudeのリスト],
                "speed":     [GPS Speedのリスト],
                "video_path": [動画ファイルのパス] }

        Returns:
            pandas.DataFrame: 重複を削除したGPSデータのDataFrame
        """
        gps_df = gps_df.drop_duplicates()

        save_dir_path = os.path.join(self.info_save_dir, "removed_duplicates")
        os.makedirs(save_dir_path, exist_ok=True)
        save_file_path = os.path.splitext(os.path.basename(self.current_video_path))[0]
        save_file_path = os.path.join(save_dir_path, save_file_path + ".csv")
        gps_df.to_csv(
            save_file_path,
            columns=["date_time", "latitude", "longitude", "speed", "video_path"],
            index=False,
        )

        return gps_df

    def interpolate_gps_data(self, gps_df, sampling_rate=15):
        """
        GPSデータを線形補間する

        Args:
            gps_df (pandas.DataFrame): GPSデータを格納したDataFrame
                { "date_time": [GPS Date/Timeのリスト],
                "latitude":  [GPS Latitudeのリスト],
                "longitude": [GPS Longitudeのリスト],
                "speed":     [GPS Speedのリスト],
                "video_path": [動画ファイルのパス] }
            sampling_rate (int): サンプリングレート（1秒あたりの補間点数）

        Returns:
            pandas.DataFrame: 補間したGPSデータのDataFrame
        """
        # 入力のDataFrameがスライスの場合に備え、元データと影響を受けないようコピーを作成
        gps_df = gps_df.copy()

        # "date_time"列の文字列をdatetime型に変換（指定フォーマットに基づく）
        gps_df["date_time"] = pd.to_datetime(
            gps_df["date_time"], format="%Y:%m:%d %H:%M:%SZ"
        )
        # datetime型の"date_time"列に+9時間を加算
        gps_df["date_time"] = gps_df["date_time"] + pd.Timedelta(hours=9)

        # "date_time"でソートして時系列順に並べ替え
        gps_df.sort_values("date_time", inplace=True)
        # "speed"列の値を数値型に変換（変換できない場合はエラーを投げる）
        gps_df["speed"] = pd.to_numeric(gps_df["speed"], errors="raise")

        # 度分秒（DMS）表記を10進法表記に変換するヘルパー関数
        def dms_to_decimal(dms_str):
            # 正規表現パターンで "deg", "'", """ と方向(N,S,E,W) をパース
            pattern = r"(\d+(?:\.\d+)?)\s*deg\s*(\d+(?:\.\d+)?)\'\s*(\d+(?:\.\d+)?)\"*\s*([NSEW])"
            match = re.search(pattern, dms_str)
            if match:
                # 度、分、秒、方向を取り出し、10進法の座標に変換する
                degrees = float(match.group(1))
                minutes = float(match.group(2))
                seconds = float(match.group(3))
                direction = match.group(4)
                # 総合して10進法の座標を計算
                decimal = degrees + minutes / 60 + seconds / 3600
                # 南または西の場合は符号を反転
                if direction in ["S", "W"]:
                    decimal = -decimal
                return decimal
            else:
                # 正規表現にマッチしなければエラーを発生させる
                raise ValueError("Invalid coordinate format: " + dms_str)

        # 各"latitude"と"longitude"の文字列を10進法形式に変換
        gps_df["latitude"] = gps_df["latitude"].apply(dms_to_decimal)
        gps_df["longitude"] = gps_df["longitude"].apply(dms_to_decimal)

        # 補間の基準となる最初の日時を取得
        t0 = gps_df["date_time"].iloc[0]
        # 各時刻との差分（秒）を計算して"time_sec"列に追加
        gps_df["time_sec"] = (gps_df["date_time"] - t0).dt.total_seconds()

        # 補間時刻の間隔（秒）を計算。sampling_rateは1秒あたりの点数
        delta_t = 1.0 / sampling_rate
        # 補間の最終時刻（秒）を取得
        final_time = gps_df["time_sec"].iloc[-1]
        # 0秒からfinal_timeまで、delta_t間隔で新しい時系列の秒数リストを作成
        new_time_sec = np.arange(0, final_time + delta_t, delta_t)
        # 各秒数に対して、基準時刻t0を足して新しいdatetimeのリストを作成
        new_date_time = [t0 + pd.Timedelta(seconds=float(s)) for s in new_time_sec]

        # 線形補間により新しい緯度、経度、速度を算出
        new_latitude = np.interp(new_time_sec, gps_df["time_sec"], gps_df["latitude"])
        new_longitude = np.interp(new_time_sec, gps_df["time_sec"], gps_df["longitude"])
        new_speed = np.interp(new_time_sec, gps_df["time_sec"], gps_df["speed"])

        # 補間した各値をDataFrameにまとめる
        interpolated_df = pd.DataFrame(
            {
                "date_time": new_date_time,
                "latitude": new_latitude,
                "longitude": new_longitude,
                "speed": new_speed,
                "video_path": self.current_video_path,  # 現在の動画ファイルパスを追加
            }
        )

        # 補間データを保存するディレクトリのパスを作成
        save_dir_path = os.path.join(self.info_save_dir, "interpolated")
        # 保存先ディレクトリが存在しない場合は作成
        os.makedirs(save_dir_path, exist_ok=True)
        # 保存ファイル名は動画ファイルのベース名から拡張子を除いたものを使用
        save_file_path = os.path.splitext(os.path.basename(self.current_video_path))[0]
        save_file_path = os.path.join(save_dir_path, save_file_path + ".csv")
        # 補間データをCSVファイルとして保存；指定カラムのみを出力し、indexは含めない
        interpolated_df.to_csv(
            save_file_path,
            columns=["date_time", "latitude", "longitude", "speed", "video_path"],
            index=False,
        )

        # 補間したDataFrameを返す
        return interpolated_df