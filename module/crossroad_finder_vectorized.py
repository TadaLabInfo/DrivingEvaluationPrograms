import numpy as np
import pandas as pd
import math
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
import os

class CrossRoadFinderVectorized:
    """
    交差点を見つけるクラス（ベクトル化版）
    道路ネットワークデータと時系列整列済みのGPSデータを用いて、交差点IDを付与します。
    
    検出基準は50m以内に交差点がある場合とし、
    その際、GPSデータの前後に extension_distance (単位:m) 分を拡張してIDを付与します。
    """
    def __init__(self, dir_path, road_network_path, info_path, threshold=50, extension_distance=150, info_save_dir=None):
        """
        コンストラクタ
        
        Parameters
        ----------
            dir_path : str
                GPSデータを保存するディレクトリのベースディレクトリ
            road_network_path : str
                道路ネットワークCSVファイルのパス
            info_path : pandas.DataFrame or str
                GPSデータ（DataFrameまたはCSVパス）
            threshold : float
                交差点検出のしきい値（単位: m、例: 50）
            extension_distance : float
                前後に拡張する距離（単位: m、例: 100）
            info_save_dir : str, optional
                保存先ディレクトリ（指定がなければ dir_path + "_gps"）
        """
        self.road_df = pd.read_csv(road_network_path)
        
        if isinstance(info_path, pd.DataFrame):
            self.gps_df = info_path
        else:
            self.gps_df = pd.read_csv(info_path)
            
        self.threshold = threshold
        self.extension_distance = extension_distance

        # GPSの緯度・経度をNumPy配列（度）として取得
        self.gps_lats = self.gps_df['latitude'].values
        self.gps_lons = self.gps_df['longitude'].values
        
        if info_save_dir is None:
            self.gps_save_dir = dir_path + "_gps"
        else:
            self.gps_save_dir = info_save_dir
        
    def __call__(self):
        self.gps_df['objectid'] = ""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("道路ネットワークデータより交差点の検出中...", total=len(self.road_df))
            
            # 各道路レコードについて処理
            for _, road in self.road_df.iterrows():
                self.assign_crossroad_objectid(road)
                progress.advance(task)
                
            progress.update(task, description="[green]道路ネットワークデータより交差点の検出中...完了")
                
        # 保存パス生成
        gps_save_path = os.path.join(self.gps_save_dir, 'gps_data_with_objectid.csv')
        # 保存
        self.gps_df.to_csv(gps_save_path, index=False)
        return self.gps_df
        
        

    def haversine_vectorized(self, lon1, lat1, lons2, lats2):
        """
        1点（lon1,lat1）とGPS全点との距離を一括計算（haversineの公式、単位：メートル）
          lon1,lat1: スカラー（道路側）　lons2,lats2: 配列（GPS側）
        """
        # スカラー側の座標はラジアンに変換
        lon1, lat1 = map(math.radians, [lon1, lat1])
        # GPS側の座標をラジアンに変換
        lons2 = np.radians(lons2)
        lats2 = np.radians(lats2)
        
        dlon = lons2 - lon1
        dlat = lats2 - lat1
        a = np.sin(dlat/2)**2 + math.cos(lat1) * np.cos(lats2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371000  # 地球の半径（m）
        return c * r
    
    def haversine(self, lon1, lat1, lon2, lat2):
        """
        2点間のhaversine距離を計算（単位：m）
        """
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000
        return c * r

    def assign_crossroad_objectid_(self, road):
        """
        ベクトル化により、各GPS点と道路点のhaversine距離を計算し、
        距離が閾値以内の場合にGPSデータに交差点(objectid)を付与します。
        
        Parameters
        ----------
        road: 道路データ（Series）
        
        Returns
        -------
        gps_df: 交差点IDを付与したGPSデータ（DataFrame）
        """
        road_lon = road['longitude']
        road_lat = road['latitude']
        road_objectid = str(road['objectid'])
        
        # 各道路点について、GPS全体との距離を一括計算
        distances = self.haversine_vectorized(road_lon, road_lat, self.gps_lons, self.gps_lats)
        # 閾値以内のGPSレコードのインデックスを抽出
        idxs = np.where(distances <= self.threshold)[0]
        
        if not len(idxs) == 0:
            print(idxs)
        
        for idx in idxs:
            current_val = self.gps_df.at[idx, 'objectid']
            if current_val == "":
                self.gps_df.at[idx, 'objectid'] = road_objectid
            else:
                # 既に別のobjectidが設定されている場合、重複しなければ連結する
                id_list = current_val.split(":")
                if road_objectid not in id_list:
                    self.gps_df.at[idx, 'objectid'] = current_val + ":" + road_objectid
                    
    def assign_crossroad_objectid(self, road):
        """
        各道路の交差点（road）に対して、まずGPS点のうち50m以内に存在する検出点群があるか確認し、
        そのグループに対して、前後self.extension_distance mずつ拡張してobjectidを付与します。
        
        前後self.extension_distance mの拡張は、GPSデータが時系列に整列していることを前提に、隣接点間のhaversine距離を累積することで実現します。
        """
        road_lon = road['longitude']
        road_lat = road['latitude']
        road_objectid = str(road['objectid'])
        
        # すべてのGPS点との距離を算出
        distances = self.haversine_vectorized(road_lon, road_lat, self.gps_lons, self.gps_lats)
        # 50m以内のGPS点インデックス（交差点検出用）
        detection_idxs = np.where(distances <= self.threshold)[0]
        
        if len(detection_idxs) == 0:
            return
        
        # 検出されたインデックス群は時系列順に並んでいると仮定
        # 連続するインデックスごとにグループ化（例： [10,11,12, 20,21] の場合、グループ[10,11,12]と[20,21] に分ける）
        groups = []
        curr_group = [detection_idxs[0]]
        for idx in detection_idxs[1:]:
            if idx == curr_group[-1] + 1:
                curr_group.append(idx)
            else:
                groups.append(curr_group)
                curr_group = [idx]
        groups.append(curr_group)
        
        # GPSデータはtime（もしくは時系列順で）が整列済みと仮定
        # 各グループについて、前後self.extension_distance m（走行距離の累積）を計算して拡張します。
        for group in groups:
            group_start = group[0]
            group_end = group[-1]
            
            # 前方（過去方向）self.extension_distance mの拡張：グループの先頭から前へ
            cum_dist = 0.0
            new_start = group_start
            i = group_start
            while i > 0:
                d = self.haversine(
                    self.gps_df.at[i, 'longitude'],
                    self.gps_df.at[i, 'latitude'],
                    self.gps_df.at[i-1, 'longitude'],
                    self.gps_df.at[i-1, 'latitude']
                )
                if cum_dist + d < self.extension_distance:
                    cum_dist += d
                    new_start = i - 1
                    i -= 1
                else:
                    break
                    
            # 後方（未来方向）self.extension_distance mの拡張：グループの末尾から後ろへ
            cum_dist = 0.0
            new_end = group_end
            i = group_end
            last_idx = self.gps_df.index[-1]
            while i < last_idx:
                d = self.haversine(
                    self.gps_df.at[i, 'longitude'],
                    self.gps_df.at[i, 'latitude'],
                    self.gps_df.at[i+1, 'longitude'],
                    self.gps_df.at[i+1, 'latitude']
                )
                if cum_dist + d < self.extension_distance:
                    cum_dist += d
                    new_end = i + 1
                    i += 1
                else:
                    break
                    
            # new_start～new_end のGPS点に対して objectid を付与
            for idx in range(new_start, new_end + 1):
                current_val = self.gps_df.at[idx, 'objectid']
                if current_val == "":
                    self.gps_df.at[idx, 'objectid'] = road_objectid
                else:
                    id_list = current_val.split(":")
                    if road_objectid not in id_list:
                        self.gps_df.at[idx, 'objectid'] = current_val + ":" + road_objectid

# if __name__ == '__main__':
#     # ファイルパスは必要に応じて調整してください
#     road_network_csv = 'aichi-network4.csv'
    
#     gps_csv_dir = 'ANJS001_gps'
#     gps_csv_file = 'combined_interpolated_gps_data.csv'
    
#     gps_csv = os.path.join(gps_csv_dir, gps_csv_file)
    
#     finder = CrossRoadFinder_Vectorized(gps_csv_dir, road_network_csv, gps_csv, threshold=50)
#     gps_with_objectid = finder()
#     # 結果をCSVファイルに出力（任意）
#     gps_save_path = os.path.join(gps_csv_dir, 'gps_data_with_objectid_vectorized.csv')
#     gps_with_objectid.to_csv(gps_save_path, index=False)
#     print(gps_with_objectid.head())