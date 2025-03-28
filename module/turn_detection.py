import glob
import pandas as pd
import math
import numpy as np
import os
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
    また、判定結果に基づいてCSVファイルのリネームも行います。

    Attributes:
        dir_path (str): GPS CSV ファイルが格納されたディレクトリのパス。
        net_path (str): ネットワーク情報CSVファイルのパス。
        gps_csvs (list): 指定ディレクトリ内の全てのGPS CSVファイルパスのリスト。
        net_df (pandas.DataFrame): ネットワーク情報のデータフレーム。
        logger (logging.Logger): ロギング用オブジェクト。
        distance_thresh (float): 交差点中心からの判定用距離（単位：メートル、例：50）。
    Methods:
        __init__(self, dir_path, net_path, distance_thresh=50): 初期化。
        __call__(self): main() メソッドを呼び出すためのオーバーロード。
        haversine(self,...): 2点間の距離(km)を計算する。
        angle_between(self,...): 2つのベクトル間の角度(度)を計算する。
        cross_z(self,...): 2次元ベクトルの外積Z成分を計算する。
        turn_judgment(self,...): 中心点と進入・退出点から進行方向を判定する。
        get_links(self, net_rec): ネットワークレコードから有効なリンク情報を抽出。
        get_nearest_link(self, pt, links): 指定点ptに最も近いリンク情報を返す。
        process_csv(self, gps_csv): 1ファイル分のGPSデータを読み込み処理を実行する。
        main(self): 全てのGPS CSVファイルに対してprocess_csvを実行する。
    """

    def __init__(self, dir_path, net_path, distance_thresh=50):
        self.dir_path = dir_path
        self.net_path = net_path
        # distance_threshはメートル単位。haversineはkmを返すので換算して保持する
        self.distance_thresh = distance_thresh / 1000.0  # 単位：km

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
        links = []
        for i in range(1, 7):
            lon_col = f"longitude{i}"
            lat_col = f"latitude{i}"
            link_col = f"link{i}"
            if lon_col in net_rec and lat_col in net_rec:
                lon_val = net_rec[lon_col]
                lat_val = net_rec[lat_col]
                if not (pd.isna(lon_val) or pd.isna(lat_val)):
                    links.append({"link": net_rec.get(link_col, i), "lon": lon_val, "lat": lat_val})
        return links

    def get_nearest_link(self, pt, links):
        dists = []
        for link in links:
            d = self.haversine(pt[0], pt[1], link["lon"], link["lat"])
            dists.append(d)
        min_index = np.argmin(dists)
        return links[min_index], dists[min_index]

    def select_point_by_distance(self, gps_df, center, target_range, idx_start, idx_end):
        """
        gps_dfのidx_start～idx_end の範囲から、交差点中心からの距離がtarget_rangeに最も近いGPS点
        （行データ）を返す。
        """
        subset = gps_df.iloc[idx_start:idx_end+1].copy()
        # 各GPS点とcenterとの距離(km)を算出
        subset["dist"] = subset.apply(lambda row: self.haversine(center[0], center[1],
                                                                  row["longitude"], row["latitude"]), axis=1)
        # target_rangeと距離の差が最小となる行を探す
        diff = (subset["dist"] - target_range).abs()
        best_idx = diff.idxmin()
        return subset.loc[best_idx]

    def process_csv(self, gps_csv):
        gps_df = pd.read_csv(gps_csv)
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

        # 交差点中心から各GPS点の距離を計算して、最短のindexを取得
        distances = gps_df.apply(lambda row: self.haversine(center[0], center[1],
                                                            row["longitude"], row["latitude"]), axis=1)
        min_idx = distances.idxmin()

        # 進入点: 交差点に入る前（0～min_idx）の中から、中心から指定距離(self.distance_thresh)に近い点を採用
        if min_idx > 0:
            inbound_row = self.select_point_by_distance(gps_df, center, self.distance_thresh, 0, min_idx)
        else:
            inbound_row = gps_df.iloc[0]

        # 退出点: 交差点通過後（min_idx～最終行）の中から、中心から指定距離に近い点を採用
        if min_idx < len(gps_df) - 1:
            outbound_row = self.select_point_by_distance(gps_df, center, self.distance_thresh, min_idx, len(gps_df)-1)
        else:
            outbound_row = gps_df.iloc[-1]

        in_pt = (inbound_row["longitude"], inbound_row["latitude"])
        out_pt = (outbound_row["longitude"], outbound_row["latitude"])
        
        # 進入リンク・退出リンクの判定
        in_link, in_dist = self.get_nearest_link(in_pt, links)
        out_link, out_dist = self.get_nearest_link(out_pt, links)
        
        # もし進入リンクと退出リンクが同じ場合は想定外の動作なのでloggingで出力
        if in_link["link"] == out_link["link"]:
            self.logger.warning(f"予期せぬ動作: objectid {objid} のファイル {gps_csv} で進入リンクと退出リンクが同じです。distance_threshを変更すると解決するかもしれません。")

        base_fname = os.path.splitext(os.path.basename(gps_csv))[0]

        stoplink_field = net_rec.get("stoplink", "")
        stop_present = str(in_link["link"]) in ("" if pd.isna(stoplink_field) else str(stoplink_field))
        stop_tag = "stop1" if stop_present else "stop0"
        
        signal_present = int(net_rec.get("sig", 0)) == 1
        sig_tag = "sig1" if signal_present else "sig0"
        
        turn, angle_val = self.turn_judgment(
            center,
            (in_link["lon"], in_link["lat"]),
            (out_link["lon"], out_link["lat"])
        )
        turn_tag = {"直進": "STR", "左折": "LFT", "右折": "RGT"}.get(turn, "UKN")
        
        new_fname = f"{base_fname}_{stop_tag}_{sig_tag}_{turn_tag}_link_{in_link['link']}_to_{out_link['link']}.csv"
        new_path = os.path.join(os.path.dirname(gps_csv), new_fname)
        os.rename(gps_csv, new_path)

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
            task = progress.add_task("右左折を判定中...", total=len(self.gps_csvs))
            for gps_csv in self.gps_csvs:
                self.process_csv(gps_csv)
                progress.update(task, advance=1)
            progress.update(task, description="[green]右左折を判定中...完了")