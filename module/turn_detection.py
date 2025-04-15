import glob
import pandas as pd
import math
import numpy as np
import os
import json
import logging
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

class TurnDetector:
    """
    TurnDetector クラスは、GPS データとネットワーク情報を利用して車両の進行方向（直進、左折、右折）を判定します。
    また、判定結果に基づいてCSVファイル内部にメタデータを付与します。
    ファイル名自体は変更せず、内部の先頭行にメタデータ(JSON)のコメント行を追加します。

    Attributes:
        dir_path (str): GPS CSV ファイルが格納されたディレクトリのパス。
        net_path (str): ネットワーク情報CSVファイルのパス。
        gps_csvs (list): 指定ディレクトリ内の全てのGPS CSVファイルパスのリスト。
        net_df (pandas.DataFrame): ネットワーク情報のデータフレーム。
        logger (logging.Logger): ロギング用オブジェクト。
        threshold_m (float): 入口・退出判定の距離閾値（メートル単位、例：10）。
    Methods:
        __init__(self, dir_path, net_path, threshold_m=10): 初期化。
        __call__(self): main() メソッドを呼び出すためのオーバーロード。
        haversine(self,...): 2点間の距離(km)を計算する。
        angle_between(self,...): 2つのベクトル間の角度(度)を計算する。
        cross_z(self,...): 2次元ベクトルの外積Z成分を計算する。
        turn_judgment(self,...): 中心点と進入・退出点から進行方向を判定する。
        get_links(self, net_rec): ネットワークレコードから有効なリンク情報を抽出（各リンクのroadid, width, lane含む）。
        detect_entry_exit(self, gps_df, links, threshold_m): GPS全レコードを走査して、各リンク座標との距離が閾値以内の最初/最後のレコードを入口・退出とする。
        process_csv(self, gps_csv): 1ファイル分のGPSデータを処理し、ファイル先頭にメタデータを埋め込む。
        main(self): 全てのGPS CSVファイルに対してprocess_csvを実行する。
    """
    def __init__(self, dir_path, net_path, threshold_m=10):
        self.dir_path = dir_path
        self.net_path = net_path
        self.threshold_m = threshold_m  # 単位：m

        self.gps_csvs = glob.glob(os.path.join(dir_path, "*", "*.csv"))
        self.net_df = pd.read_csv(net_path)

        self.logger = logging.getLogger("TurnDetector")

    def __call__(self):
        self.main()

    # haversine距離 (km) を計算する関数
    def haversine(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a)) 
        r = 6371
        return c * r

    def angle_between(self, v1, v2):
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0
        cosine = dot / (norm1 * norm2)
        cosine = max(min(cosine, 1), -1)
        angle = math.degrees(math.acos(cosine))
        return angle

    def cross_z(self, v1, v2):
        return v1[0]*v2[1] - v1[1]*v2[0]

    def turn_judgment(self, center, in_pt, out_pt, straight_thresh=20):
        v_in = np.array([in_pt[0] - center[0], in_pt[1] - center[1]])  # 進入ベクトル
        v_out = np.array([out_pt[0] - center[0], out_pt[1] - center[1]])  # 退出ベクトル
        ang = self.angle_between(v_in, v_out)
        if ang < straight_thresh or abs(ang - 180) < straight_thresh:
            return "直進", ang
        cz = self.cross_z(v_in, v_out)
        if cz > 0:
            return "右折", ang
        else:
            return "左折", ang

    def get_links(self, net_rec):
        """
        ネットワークレコードから有効なリンク情報を抽出する。
        各リンクについて、longitudeX, latitudeX, roadidX, linkX, widthX, laneX の情報を含める。
        """
        links = []
        for i in range(1, 7):
            lon_col = f"longitude{i}"
            lat_col = f"latitude{i}"
            link_col = f"link{i}"
            if lon_col in net_rec and lat_col in net_rec:
                lon_val = net_rec[lon_col]
                lat_val = net_rec[lat_col]
                if not (pd.isna(lon_val) or pd.isna(lat_val)):
                    roadid = net_rec.get(f"roadid{i}", None)
                    width = net_rec.get(f"width{i}", None)
                    lane = net_rec.get(f"lane{i}", None)
                    links.append({
                        "link": str(net_rec.get(link_col, i)),
                        "lon": lon_val,
                        "lat": lat_val,
                        "roadid": roadid,
                        "width": width,
                        "lane": lane
                    })
        return links

    def detect_entry_exit(self, gps_df, links, threshold_m):
        """
        GPSデータ全レコードから、各リンク座標との距離が threshold_m (メートル) 以下となるレコードを探索する。
        順方向に探索して最初にヒットしたものを入口、逆方向に探索して最初にヒットしたものを退出とする。
        """
        threshold_km = threshold_m / 1000.0
        entry_link = None
        # 順方向探索（入口）
        for idx, row in gps_df.iterrows():
            point = (row["longitude"], row["latitude"])
            candidates = []
            for link in links:
                d = self.haversine(point[0], point[1], link["lon"], link["lat"])
                if d <= threshold_km:
                    candidates.append((link, d))
            if candidates:
                entry_link = min(candidates, key=lambda x: x[1])[0]
                break

        exit_link = None
        # 逆方向探索（退出）
        for idx in gps_df.index[::-1]:
            row = gps_df.loc[idx]
            point = (row["longitude"], row["latitude"])
            candidates = []
            for link in links:
                d = self.haversine(point[0], point[1], link["lon"], link["lat"])
                if d <= threshold_km:
                    candidates.append((link, d))
            if candidates:
                exit_link = min(candidates, key=lambda x: x[1])[0]
                break

        return entry_link, exit_link

    def process_csv(self, gps_csv):
        # pandasはcomment="#"により先頭のメタデータ行を読み飛ばす
        gps_df = pd.read_csv(gps_csv, comment="#")
        objid = gps_df["objectid"].iloc[0]
        net_rec = self.net_df[self.net_df["objectid"] == objid]
        if net_rec.empty:
            print("対象objectidのネットワークレコードが見つかりません。ファイル:", gps_csv)
            return
        net_rec = net_rec.iloc[0]
        center = (net_rec["longitude"], net_rec["latitude"])
        links = self.get_links(net_rec)
        if not links:
            print("ネットワークレコードに有効なリンク座標が見つかりません。ファイル:", gps_csv)
            return

        # GPS全レコードから、閾値内での入口・退出リンクを検出
        entry_link, exit_link = self.detect_entry_exit(gps_df, links, threshold_m=self.threshold_m)
        if entry_link is None or exit_link is None:
            print("GPSデータから入口または退出のリンクが検出できませんでした。ファイル:", gps_csv)
            return

        if entry_link["link"] == exit_link["link"]:
            self.logger.warning(f"予期せぬ動作: objectid {objid} のファイル {gps_csv} で入口リンクと退出リンクが同じです。threshold_mを変更すると解決するかもしれません。")

        # 進行方向の判定
        turn, angle_val = self.turn_judgment(
            center,
            (entry_link["lon"], entry_link["lat"]),
            (exit_link["lon"], exit_link["lat"])
        )
        # 日本語の進行方向を英語表記に変換するマップ
        turn_map = {"直進": "STR", "右折": "RGT", "左折": "LFT"}
        turn_en = turn_map.get(turn, "UKN")
        
        # stoplinkのフィールドから、進入方向に一時停止があるかどうかを判定（0: なし、1: あり）
        stoplink_field = net_rec.get("stoplink", "")
        if pd.isna(stoplink_field):
            stop_value = 0
        else:
            stop_value = 1 if str(entry_link["link"]) in str(stoplink_field) else 0

        # メタデータとして記録する情報を辞書にまとめる（angleは削除）
        metadata = {
            "objectid": objid,
            "center": {"lon": center[0], "lat": center[1]},
            "stop": stop_value,
            "sig": int(net_rec.get("sig", 0)),
            "turn": turn_en,
            "entry_link": {
                "link": entry_link["link"],
                "roadid": entry_link["roadid"],
                "width": entry_link["width"],
                "lon": entry_link["lon"],
                "lat": entry_link["lat"]
            },
            "exit_link": {
                "link": exit_link["link"],
                "roadid": exit_link["roadid"],
                "width": exit_link["width"],
                "lon": exit_link["lon"],
                "lat": exit_link["lat"]
            }
        }

        # json.dumpsでdefault=strを設定：シリアライズできない型を文字列に変換する
        metadata_comment = "#METADATA: " + json.dumps(metadata, default=str) + "\n"
        with open(gps_csv, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 既に先頭に"#METADATA:"がある場合は削除
        if lines and lines[0].startswith("#METADATA:"):
            lines = lines[1:]
        with open(gps_csv, 'w', encoding='utf-8') as f:
            f.write(metadata_comment)
            f.writelines(lines)

    def main(self):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("GPSデータの処理中...", total=len(self.gps_csvs))
            for gps_csv in self.gps_csvs:
                self.process_csv(gps_csv)
                progress.update(task, advance=1)
            progress.update(task, description="[green]処理完了")