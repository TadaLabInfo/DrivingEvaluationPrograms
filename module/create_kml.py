import pandas as pd
import simplekml
import glob
import os
import json
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
    
    def get_metadata_from_csv(self, csv_path):
        """
        CSVファイルの先頭行に記載されるメタデータコメント行(例: "#METADATA: {...}")を取得して解析する。
        メタデータが見つからなければNoneを返す。
        """
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        if first_line.startswith("#METADATA:"):
            json_str = first_line[len("#METADATA:"):].strip()
            try:
                metadata = json.loads(json_str)
                return metadata
            except json.JSONDecodeError:
                return None
        return None
        
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
                # CSVファイルのあるディレクトリ・ベース名を取得
                file_dir_path = os.path.dirname(file_path)
                base_name = os.path.splitext(os.path.basename(file_path))[0]

                # まず、メタデータを取得（存在しなければスキップ）
                metadata = self.get_metadata_from_csv(file_path)
                if metadata is None:
                    print("メタデータが見つかりません。スキップ:", file_path)
                    progress.advance(task)
                    continue

                # pandas.read_csvで先頭のコメント行を除く（comment="#"を指定）
                df = pd.read_csv(file_path, comment="#")
                # GPSの軌跡(LineString)用の座標リスト
                line_coords = list(zip(df['longitude'], df['latitude']))
                
                # メタデータから objectid を取得し、int型に変換
                objectid = metadata.get("objectid", None)
                if objectid is None:
                    print("objectidが不足しています。スキップ:", file_path)
                    progress.advance(task)
                    continue
                try:
                    objectid = int(objectid)
                except ValueError:
                    print("objectidの変換に失敗しました。スキップ:", file_path)
                    progress.advance(task)
                    continue

                # ネットワーク情報の objectid 列も int 型に変換して比較
                network_df_clip = self.network_df[self.network_df['objectid'].astype(int) == objectid]
                if network_df_clip.empty:
                    print("ネットワーク情報が見つかりません。objectid:", objectid)
                    progress.advance(task)
                    continue
                # 交差点の座標（ネットワークデータの中心）
                intersection = (network_df_clip['longitude'].iloc[0], network_df_clip['latitude'].iloc[0])
                
                # KMLオブジェクトの作成
                kml = simplekml.Kml()

                # 軌跡(LineString)
                ls = kml.newlinestring(name="Trajectory", description="GPS Trajectory")
                ls.coords = line_coords
                ls.style.linestyle.color = simplekml.Color.rgb(255, 0, 0)
                ls.style.linestyle.width = 3
                ls.altitudemode = simplekml.AltitudeMode.clamptoground

                start_coord = line_coords[0]
                end_coord = line_coords[-1]

                start_point = kml.newpoint(name="Start", description="Start of trajectory")
                start_point.coords = [start_coord]
                start_point.style.iconstyle.color = simplekml.Color.rgb(255, 255, 0)
                start_point.style.iconstyle.scale = 1.2
                start_point.altitudemode = simplekml.AltitudeMode.clamptoground

                end_point = kml.newpoint(name="End", description="End of trajectory")
                end_point.coords = [end_coord]
                end_point.style.iconstyle.color = simplekml.Color.rgb(255, 165, 0)
                end_point.style.iconstyle.scale = 1.2
                end_point.altitudemode = simplekml.AltitudeMode.clamptoground

                # 交差点
                turn = metadata.get("turn", "UKN")
                pnt = kml.newpoint(name=f"Intersection ({turn})", description="Intersection")
                pnt.coords = [intersection]
                pnt.style.iconstyle.color = simplekml.Color.rgb(0, 0, 255)
                pnt.style.iconstyle.scale = 1.2
                pnt.altitudemode = simplekml.AltitudeMode.clamptoground

                # 入口リンク(Entry)のポイント
                entry = metadata.get("entry_link", {})
                entry_link = entry.get("link", "NA")
                entry_lon = entry.get("lon", None)
                entry_lat = entry.get("lat", None)
                if entry_lon is not None and entry_lat is not None:
                    pnt_entry = kml.newpoint(name=f"Entry Link {entry_link}",
                                             description=f"RoadID: {entry.get('roadid')} Width: {entry.get('width')}")
                    pnt_entry.coords = [(entry_lon, entry_lat)]
                    pnt_entry.style.iconstyle.color = simplekml.Color.rgb(0, 255, 0)
                    pnt_entry.style.iconstyle.scale = 1.2
                    pnt_entry.altitudemode = simplekml.AltitudeMode.clamptoground

                # 退出リンク(Exit)のポイント
                exit_info = metadata.get("exit_link", {})
                exit_link = exit_info.get("link", "NA")
                exit_lon = exit_info.get("lon", None)
                exit_lat = exit_info.get("lat", None)
                if exit_lon is not None and exit_lat is not None:
                    pnt_exit = kml.newpoint(name=f"Exit Link {exit_link}",
                                            description=f"RoadID: {exit_info.get('roadid')} Width: {exit_info.get('width')}")
                    pnt_exit.coords = [(exit_lon, exit_lat)]
                    pnt_exit.style.iconstyle.color = simplekml.Color.rgb(0, 255, 0)
                    pnt_exit.style.iconstyle.scale = 1.2
                    pnt_exit.altitudemode = simplekml.AltitudeMode.clamptoground

                # ドキュメント全体へextended dataとしてメタデータを埋め込む
                import json
                meta_str = json.dumps(metadata, ensure_ascii=False, indent=2)
                if kml.document.extendeddata is None:
                    kml.document.extendeddata = simplekml.ExtendedData()
                kml.document.extendeddata.newdata(name="metadata", value=meta_str)

                # 保存：CSVと同じベース名に拡張子.kmlを付与
                kml_file_path = os.path.join(file_dir_path, base_name + ".kml")
                kml.save(kml_file_path)
                progress.advance(task)
            progress.update(task, description="[green]KMLファイルの作成完了")
            
# テスト用
# road_network_csv = "aichi-network4.csv"
# dir_path = "ANJS001_temp3_info/objectid_csvs"
# create_kml = CreateKML(dir_path, road_network_csv)
# create_kml()