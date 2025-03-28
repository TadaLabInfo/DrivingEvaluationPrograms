"""
【連結動画からのGPS区間切り出しサンプル】
・GPS CSV の各レコードは絶対時刻 (date_time) と動画ファイルパス (video_path) を持ち，
  各動画の名前から撮影開始時刻を抽出します。
・各動画は必ず連続撮影されている前提で，まず全動画を連結し，
  先頭動画の開始時刻を基準に GPS の絶対時刻→全体動画内におけるオフセットを計算します。
・交差点中心から最も近いGPSレコード（idx0）を基準に前後 accum_distance [m] 分の区間を
  連結動画全体内のグローバルオフセットで切り出します。
  accum_distance が -1 の場合は、GPSデータ全体の範囲を切り出します。
"""

import logging
import pandas as pd
import datetime
from geopy.distance import distance
import os
import re
import glob
import cv2
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

class CreateMovie:
    def __init__(self, dir_path, network_csv, accum_distance=-1.0):
        self.accum_distance = accum_distance
        self.net_df = pd.read_csv(network_csv)
        # 指定ディレクトリ内の全てのCSVファイルを取得（サブディレクトリも含む）
        self.csv_file_paths = glob.glob(os.path.join(dir_path, "*", "*.csv"))
        self.logger = logging.getLogger("CreateMovie")
        
    def __call__(self):
        self.create()
        
    # 日時文字列をタイムスタンプに変換する関数（修正済み）
    def parse_date_time(self, date_str):
        if pd.isna(date_str):
            return pd.NaT
        # 数値であっても文字列に変換する
        date_str = str(date_str)
        if "." in date_str:  # ミリ秒が含まれる場合
            return pd.to_datetime(date_str, format="%Y-%m-%d %H:%M:%S.%f%z")
        else:
            return pd.to_datetime(date_str, format="%Y-%m-%d %H:%M:%S%z")
        
    # 動画ファイル名から「YYMMDD-HHMMSS」の部分を抽出する関数
    def parse_video_start_time(self, video_filename):
        m = re.search(r'(\d{6}-\d{6})', video_filename)
        if m:
            dt_str = m.group(1)  # 例："241023-160807"
            video_start = datetime.datetime.strptime(dt_str, "%y%m%d-%H%M%S")
            return video_start
        else:
            raise ValueError(f"動画ファイル名から開始時刻が抽出できません: {video_filename}")
        
    def create(self):
        """
        処理の流れ:
          1. 各GPSデータCSVを読み込み，date_time カラムをパース
          2. GPSデータに記載の動画ファイルパス（video_path）を基に，開始時刻でソート
          3. ソート済みの動画それぞれについて，cv2.VideoCapture で動画情報（fps, フレーム数）を取得
             → 各動画の連結後のグローバルタイム（秒）を計算するための情報をリスト化
          4. 先頭動画の開始時刻を元に，GPS各レコードの時刻を連結動画内のオフセット（秒）に変換
          5. 同じ交差点(objectid)のGPSデータから，交差点中心との距離を計算し，
             その中で交差点中心に最も近いGPSレコードを特定（idx0）
          6. accum_distance が -1 の場合は全GPS範囲を，そうでなければ前後 accum_distance m の区間を算出
          7. GPSの global_offset 値から，全体動画内での切り出し開始・終了時刻（秒）を取得
          8. 連結動画は各動画を順次読み込みながら出力動画に書き出す。各動画ファイルは，
             連結動画のグローバルタイムに合わせ，必要なフレームのみ出力する。
        """
        # CSV毎に処理する
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("連結動画の作成中...", total=len(self.csv_file_paths))
            for gps_csv in self.csv_file_paths:
                # ファイル名からディレクトリとファイル名を分割し、ファイル名は拡張子も除去する
                dir_name, file_name = os.path.split(gps_csv)
                save_video_path = os.path.join(dir_name, os.path.splitext(file_name)[0] + f"_{self.accum_distance}m.mp4")
                gps_df = pd.read_csv(gps_csv)
                # GPSデータの日時をタイムスタンプに変換
                gps_df["date_time"] = gps_df["date_time"].apply(self.parse_date_time)
            
                # 型がdatetime64でない場合は変換
                if not pd.api.types.is_datetime64_any_dtype(gps_df["date_time"]):
                    gps_df["date_time"] = pd.to_datetime(gps_df["date_time"])
        
                # 動画パスを取得し、開始時刻でソート
                video_paths = list(gps_df["video_path"].unique())
                video_paths = sorted(video_paths, key=lambda vp: self.parse_video_start_time(vp))
                
                # 各動画ファイルの情報（fps, frame数, duration）をリストに集める
                video_info_list = []
                for vp in video_paths:
                    if not os.path.exists(vp):
                        print(f"動画ファイルが存在しません: {vp}")
                        continue
                    cap = cv2.VideoCapture(vp)
                    if not cap.isOpened():
                        print(f"動画ファイルが開けません: {vp}")
                        continue
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        print(f"fps取得エラー: {vp}")
                        cap.release()
                        continue
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps
                    video_info_list.append({
                        "path": vp,
                        "fps": fps,
                        "frame_count": frame_count,
                        "duration": duration
                    })
                    cap.release()
                
                if len(video_info_list) == 0:
                    print("有効な動画ファイルが見つかりません")
                    continue
                
                # 連結動画全体のグローバルタイムは、各動画の duration の累積として計算する
                cumulative_durations = []
                cum = 0.0
                for info in video_info_list:
                    cumulative_durations.append(cum)
                    cum += info["duration"]
                total_duration = cum
                
                # 連結動画の fps, frame size は先頭動画のものを採用
                cap0 = cv2.VideoCapture(video_info_list[0]["path"])
                fps = cap0.get(cv2.CAP_PROP_FPS)
                width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap0.release()
                
                # 連結動画の開始時刻は先頭動画の撮影開始時刻とする（GPSデータのタイムゾーン情報を利用）
                first_video_start = self.parse_video_start_time(video_paths[0])
                first_video_tz = gps_df["date_time"].iloc[0].tzinfo
                first_video_start = first_video_start.replace(tzinfo=first_video_tz)
                
                # GPS各レコードに先頭動画との秒数差 (global_offset) を計算
                gps_df["global_offset"] = gps_df["date_time"].apply(lambda dt: (dt - first_video_start).total_seconds())
                
                # ここでは対象の交差点(objectid)はすべて同じと仮定し，先頭レコードの objectid を採用
                objid = gps_df.iloc[0]["objectid"]
                sub_gps = gps_df[gps_df["objectid"] == objid].sort_values("date_time").reset_index(drop=True)
                
                # 交差点ネットワークCSVから交差点中心の座標を取得
                net_row = self.net_df[self.net_df["objectid"] == objid]
                if net_row.empty:
                    print(f"交差点(objectid={objid})の情報が見つかりません")
                    continue
                inter_lat = net_row.iloc[0]["latitude"]
                inter_lon = net_row.iloc[0]["longitude"]
                inter_coord = (inter_lat, inter_lon)
                
                # 各GPSレコードと交差点中心との距離 [m] を計算
                sub_gps["dist"] = sub_gps.apply(lambda row: distance((row["latitude"], row["longitude"]), inter_coord).meters, axis=1)
                idx0 = sub_gps["dist"].idxmin()
                
                # accum_distance が -1 の場合は全範囲を使用
                if self.accum_distance == -1.0:
                    start_idx = 0
                    end_idx = sub_gps.shape[0] - 1
                else:
                    # idx0 から前方向の累積距離で切り出し開始の index を決定
                    cum_dist = 0.0
                    start_idx = idx0
                    for i in range(idx0, 0, -1):
                        d = distance((sub_gps.loc[i, "latitude"], sub_gps.loc[i, "longitude"]),
                                    (sub_gps.loc[i-1, "latitude"], sub_gps.loc[i-1, "longitude"])).meters
                        cum_dist += d
                        if cum_dist >= self.accum_distance:
                            start_idx = i-1
                            break
                    # idx0 から後方向の累積距離で切り出し終了の index を決定
                    cum_dist = 0.0
                    end_idx = idx0
                    for i in range(idx0, sub_gps.shape[0]-1):
                        d = distance((sub_gps.loc[i, "latitude"], sub_gps.loc[i, "longitude"]),
                                    (sub_gps.loc[i+1, "latitude"], sub_gps.loc[i+1, "longitude"])).meters
                        cum_dist += d
                        if cum_dist >= self.accum_distance:
                            end_idx = i+1
                            break
                
                # 全体動画内（連結動画）の切り出し開始・終了オフセット（秒）を取得
                offset_start = sub_gps.loc[start_idx, "global_offset"]
                offset_end   = sub_gps.loc[end_idx, "global_offset"]
                if offset_start < 0:
                    print("GPSデータの一部が動画開始前のため、offset_start を0に補正します。")
                    offset_start = 0.0
                if offset_end > total_duration:
                    print("GPSデータの一部が連結動画の末尾を越えているため、offset_end を総再生時間に補正します。")
                    offset_end = total_duration
                
                # print(f"切り出し範囲：{offset_start:.2f}秒 ～ {offset_end:.2f}秒 (全体動画時間: {total_duration:.2f}秒)")
                self.logger.debug(f"切り出し範囲：{offset_start:.2f}秒 ～ {offset_end:.2f}秒 (全体動画時間: {total_duration:.2f}秒)")
                
                # 出力動画ファイルの設定
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
                if not out_writer.isOpened():
                    print("VideoWriterの初期化に失敗しました")
                    continue
                
                # フォント設定（大きく右上に描画するために調整）
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2
                thickness = 3
                color = (0, 255, 0)  # デフォルトは緑色。以下で6dreplacedの値により変更します。
                margin_x, margin_y = 10, 10
                
                # 各動画ファイルから、連結動画内の対象範囲に該当するフレームを順次書き出す
                sub_task = progress.add_task("切り出し動画の作成中...", total=len(video_info_list))
                for info, seg_start in zip(video_info_list, cumulative_durations):
                    video_path = info["path"]
                    duration = info["duration"]
                    seg_global_start = seg_start
                    seg_global_end = seg_start + duration
                    # 対象区間 [offset_start, offset_end] とこの動画セグメントの重なりがあるか確認
                    if seg_global_end < offset_start:
                        # この動画は切り出し範囲より前
                        continue
                    if seg_global_start > offset_end:
                        # 既に切り出し範囲を終えている
                        break
                    
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"動画ファイルが開けません: {video_path}")
                        continue
                    
                    # このセグメント内での切り出し開始・終了（秒）
                    inner_start = max(offset_start - seg_global_start, 0.0)
                    inner_end   = min(offset_end - seg_global_start, duration)
                    # 対応するフレーム位置
                    start_frame_idx = int(inner_start * fps)
                    end_frame_idx = int(inner_end * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
                    
                    current_frame_idx = start_frame_idx
                    subsub_task = progress.add_task("動画の切り出し中...", total=end_frame_idx - start_frame_idx)
                    while current_frame_idx < end_frame_idx:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # 現在の連結動画上でのグローバルタイム（秒）を計算
                        frame_global_time = seg_global_start + (current_frame_idx / fps)
                        
                        # GPSデータ（sub_gps）の中から，現在のフレーム時刻以下の最後のレコードを取得
                        applicable = sub_gps[sub_gps["global_offset"] <= frame_global_time]
                        if not applicable.empty:
                            face_angle = applicable.iloc[-1]["face_angle"]
                            sixd_replaced = applicable.iloc[-1]["6dreplaced"]
                            # 6dreplacedが0なら緑、1なら赤でテキスト描画
                            try:
                                if int(sixd_replaced) == 0:
                                    text_color = (0, 255, 0)  # 緑
                                elif int(sixd_replaced) == 1:
                                    text_color = (0, 0, 255)  # 赤
                                else:
                                    text_color = color  # 想定外の場合はデフォルト色
                            except Exception:
                                text_color = color
                        else:
                            face_angle = ""
                            text_color = color
                        
                        # 描画するテキスト（右上に大きく表示）
                        text = f"Face: {face_angle}"
                        # テキストサイズを取得し、右上に配置するための起点座標を計算
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        # OpenCVでは座標がテキストの左下となるため、右上に表示するには
                        # x座標：フレーム幅 - テキスト幅 - margin, y座標：margin + テキスト高さ
                        pos = (frame.shape[1] - text_width - margin_x, margin_y + text_height)
                        
                        cv2.putText(frame, text, pos, font, font_scale, text_color, thickness, cv2.LINE_AA)
                        
                        out_writer.write(frame)
                        current_frame_idx += 1
                        progress.advance(subsub_task)
                    progress.remove_task(subsub_task)
                    progress.advance(sub_task)
                    cap.release()
                progress.remove_task(sub_task)
                
                out_writer.release()
                # print(f"切り出し動画を保存しました: {save_video_path}")
                self.logger.info(f"切り出し動画を保存しました: {save_video_path}")
                progress.advance(task)
            progress.update(task, description="[green]連結動画の作成中...完了")

# # テスト用コード
# # GPSデータCSV と交差点ネットワークCSV のパス（環境に合わせて調整してください）
# gps_csv_dir = r"ANJO001_temp_info\objectid_csvs"
# network_csv = "test_network4.csv"

# # GPSデータと交差点ネットワークデータから連結動画を生成し、指定距離で切り出す
# create_movie = CreateMovie(gps_csv_dir, network_csv, accum_distance=50)
# create_movie()