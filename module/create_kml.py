from calendar import c
import pandas as pd
import simplekml
from shapely.geometry import LineString
import glob
import os
import re
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

class CreateKML:
    def __init__(self, dir_path, road_network_csv):
        self.network_df = pd.read_csv(road_network_csv)
        self.file_paths = glob.glob(os.path.join(dir_path, "*", "*.csv"))

    def __call__(self):
        self.create_kml_files()
        
    def get_direction_from_file_name(self, text):
        """
        与えられた文字列(text)から、必ず3文字であるRGT, STR, LFT, UKNのいずれかを抽出する。
        例として、ファイル名 "2024-10-23_16-09-02_stop0_sig1_RGT_link_6272377_to_4108443" からは "RGT" を返す。
        """
        # アンダースコアで囲まれている3文字大文字の部分を探す。今回は3文字なので、パターンは _([A-Z]{3})_ とする。
        pattern = r'_([A-Z]{3})_'
        match = re.search(pattern, text)
        if match:
            token = match.group(1)
            return token
        return None
        
    def get_link_from_file_name(self, file_name):
        """
        ファイル名からリンク番号を抽出する。
        例として、ファイル名 "2024-10-23_16-09-02_stop0_sig1_RGT_link_6272377_to_4108443" からは "6272377" と "4108443" を返す。
        """
        # 正規表現でリンク番号を抽出
        pattern = r"link_(\d+)_to_(\d+)"
        match = re.search(pattern, file_name)
        if match:
            start_link = match.group(1)
            end_link = match.group(2)
            return start_link, end_link
        return None, None
    
    def get_coordinates_for_link(self, row, target_link):
        """
        DataFrameの1行分(row)から、link1 ~ link6の中でtarget_linkに一致する列の番号を特定し、
        その番号に対応するlongitudeX, latitudeXの値を返す。
        """
        # 今回は1～6までループ
        for i in range(1, 7):
            link_col = f"link{i}"
            # CSVの値が文字列または数値として保存されている可能性があるので、str()に変換して比較
            if str(row[link_col]) == target_link:
                lon = row[f"longitude{i}"]
                lat = row[f"latitude{i}"]
                return lon, lat
        # 一致するリンクが見つからなかった場合はNoneを返す
        return None, None

    def create_kml_files(self):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("KMLファイルの作成中...", total=len(self.file_paths))
            
            for file_path in self.file_paths:
                # file_pathより、ファイル名を除いたディレクトリパスを取得
                file_dir_path = os.path.dirname(file_path)
                # file_pathより、拡張子を除いたファイル名を取得
                file_name = os.path.splitext(os.path.basename(file_path))[0]

                df = pd.read_csv(file_path)

                # GPS情報から軌跡（LineString）を作成するための座標リストを取得
                line_coords = list(zip(df['longitude'], df['latitude']))
                
                # GPS情報からobjectidを取得
                objectid = df['objectid'].iloc[0]
                
                # 道路ネットワークデータより、objectidに対応する交差点の座標を取得
                network_df_clip = self.network_df[self.network_df['objectid'] == objectid]
                if network_df_clip.empty:
                    print("No network data for objectid:", objectid)
                    continue
                intersection = (network_df_clip['longitude'].iloc[0], network_df_clip['latitude'].iloc[0])
                # 交差点座標を設定（経度, 緯度）
                highlight_coord = intersection

                # KMLオブジェクトの作成と軌跡・ポイントの追加
                kml = simplekml.Kml()

                # 軌跡をKMLのLineStringとして追加
                ls = kml.newlinestring(name="Trajectory", description="GPS Trajectory")
                ls.coords = line_coords
                ls.style.linestyle.color = simplekml.Color.rgb(255, 0, 0)  # 赤色
                ls.style.linestyle.width = 3
                # 高度を地面に合わせるため、altitudemodeをclampToGroundに設定
                ls.altitudemode = simplekml.AltitudeMode.clamptoground

                # --- 軌跡の始点と終点のポイントを追加 ---
                start_coord = line_coords[0]
                end_coord = line_coords[-1]
                
                start_point = kml.newpoint(name="Start", description="Start of trajectory")
                start_point.coords = [start_coord]
                start_point.style.iconstyle.color = simplekml.Color.rgb(255, 255, 0)  # 黄色
                start_point.style.iconstyle.scale = 1.2
                start_point.altitudemode = simplekml.AltitudeMode.clamptoground

                end_point = kml.newpoint(name="End", description="End of trajectory")
                end_point.coords = [end_coord]
                end_point.style.iconstyle.color = simplekml.Color.rgb(255, 165, 0)  # オレンジ色
                end_point.style.iconstyle.scale = 1.2
                end_point.altitudemode = simplekml.AltitudeMode.clamptoground
                # ------------------------------------------

                # 交差点をKMLのPointとして追加
                direction = self.get_direction_from_file_name(file_name)
                pnt = kml.newpoint(name=f"CrossRoad_{direction}", description="Intersection")
                pnt.coords = [highlight_coord]
                pnt.style.iconstyle.color = simplekml.Color.rgb(0, 0, 255)  # 青色
                pnt.style.iconstyle.scale = 1.2
                # ポイントも地面に合わせる
                pnt.altitudemode = simplekml.AltitudeMode.clamptoground
                
                # 道路の隣接道路(入り)をKMLのPointとして追加
                start_link, end_link = self.get_link_from_file_name(file_name)  # 正規表現でリンク番号を抽出
                start_lon, start_lat = self.get_coordinates_for_link(network_df_clip.iloc[0], start_link)
                end_lon, end_lat = self.get_coordinates_for_link(network_df_clip.iloc[0], end_link)
                pnt1 = kml.newpoint(name=f"StartLink {start_link}", description=f"Link {start_link} coordinate")
                pnt1.coords = [(start_lon, start_lat)]
                pnt1.style.iconstyle.color = simplekml.Color.rgb(0, 255, 0)  # 緑色
                pnt1.style.iconstyle.scale = 1.2
                pnt1.altitudemode = simplekml.AltitudeMode.clamptoground
                
                # 道路の隣接道路(出)をKMLのPointとして追加
                pnt2 = kml.newpoint(name=f"EndLink {end_link}", description=f"Link {end_link} coordinate")
                pnt2.coords = [(end_lon, end_lat)]
                pnt2.style.iconstyle.color = simplekml.Color.rgb(0, 255, 0)  # 緑色
                pnt2.style.iconstyle.scale = 1.2
                pnt2.altitudemode = simplekml.AltitudeMode.clamptoground

                # KMLファイルの保存(dir_path内にファイル名.kmlで保存)
                kml_file_path = os.path.join(file_dir_path, file_name + ".kml")
                kml.save(kml_file_path)
                progress.advance(task)
            progress.update(task, description="[green]KMLファイルの作成中...完了")
            
# テスト用
# road_network_csv = "aichi-network4.csv"
# dir_path = "ANJS001_temp3_info/objectid_csvs"
# create_kml = CreateKML(road_network_csv, dir_path)
# create_kml()
