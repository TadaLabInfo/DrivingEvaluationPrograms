import os
import pandas as pd
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

class GPSSplitter:
    """
    GPSSplitter クラスは、GPS データを objectid ごとに複数の走行区間に分割し、各区間ごとに CSV ファイルとして保存するためのユーティリティです。
    中川追記: 簡単に言うと、例えばAという交差点とBという交差点を連続で走行しているとき、AからBまでの間を区間として分割し、それをCSVファイルとして保存するものです。
    Attributes:
        dir_path (str):
            GPS データの保存先となるベースディレクトリのパスです。最終的に、各 objectid のデータはこのディレクトリ内の "objectid_csvs" サブディレクトリに保存されます。
        gps_df (pandas.DataFrame):
            入力として渡された GPS データを読み込み保持する DataFrame です。gps_path に DataFrame が渡された場合はそのまま利用し、文字列（CSV ファイルパス）の場合は pandas.read_csv により読み込みます。
    Methods:
        __init__(self, dir_path: str, gps_path: Union[str, pandas.DataFrame]):
            コンストラクタ。GPS データの読み込みと保存先ディレクトリの初期設定を行います。
                dir_path (str): GPS データの出力先ディレクトリのベースパス。
                gps_path (str または pandas.DataFrame): GPS データの CSV ファイルパス、もしくは直接渡される DataFrame。
        __call__(self) -> str:
            GPS データを以下の手順で処理します：
                1. 出力ディレクトリ (dir_path/objectid_csvs) を作成する。
                2. DataFrame の "objectid" 列の欠損値や空文字に対して適切な補完を行う。
                3. "objectid" 列に ":" が含まれている場合、文字列を分割しリスト形式に変換する。
                4. pandas の explode() を利用して、複数の objectid を持つ行を個別の行に展開し、不要な空文字の行を除外する。
                5. 各ユニークな objectid ごとに以下の処理を実施する：
                   - 該当する行のみを抽出し、補助カラムを削除する。
                   - "date_time" 列を datetime 型に変換し、エラー時は NaT として扱う。
                   - 日時順にソートし、前後のレコード間の差が 30 秒以上の場合を新たな走行区間（セグメント）とみなす。
                   - 各セグメントごとに、最初のレコードの日付を用いて "YYYY-MM-DD_HH-MM-SS" 形式のファイル名を生成し、objectid ごとのサブディレクトリに CSV ファイルとして保存する。
                6. 全 objectid の処理が完了した後、出力先のディレクトリパス (csv_save_dir) を返す。
            また、データ処理の進捗表示には Rich ライブラリの Progress クラスを利用しています。
    Usage Example:
        >>> splitter = GPSSplitter("/path/to/output", "/path/to/gps.csv")
        >>> output_dir = splitter()
        >>> print("GPS データの分割が完了しました。保存先:", output_dir)
    """
    
    
    def __init__(self, dir_path, gps_path):
        """Initializer

        Args:
            dir_path (str): ディレクトリのパス（GPSデータを保存するディレクトリのベースディレクトリ）
            gps_path (_type_): GPSデータのCSVファイルパスまたはDataFrame
        """
        
        # もしgps_pathがDataFrameなら、そのまま代入
        if isinstance(gps_path, pd.DataFrame):
            self.gps_df = gps_path
        else:
            self.gps_df = pd.read_csv(gps_path)
            
        self.dir_path = dir_path
    
    def __call__(self):
        # 保存先のベースディレクトリ作成（後でobjectid毎にサブディレクトリを作成）
        csv_save_dir = os.path.join(self.dir_path, "objectid_csvs")
        self.csv_save_dir = csv_save_dir
        os.makedirs(self.csv_save_dir, exist_ok=True)

        # objectid列がNaNの場合や空文字の場合に対処
        self.gps_df["objectid"] = self.gps_df["objectid"].fillna("")

        # objectid列に":"が含まれている場合、分割してリストにする
        # 例: "2882045:2888928" → ["2882045", "2888928"]
        self.gps_df["objectid_list"] = self.gps_df["objectid"].apply(lambda x: x.split(':') if x != "" else [])

        # 行ごとに複数のobjectidがある場合、explodeで行を個別に展開
        df_exploded = self.gps_df.explode("objectid_list")

        # 空文字（objectidが空）の行は除外
        df_exploded = df_exploded[df_exploded["objectid_list"] != ""]

        # 各objectidごとに処理する
        unique_ids = df_exploded["objectid_list"].unique()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("objectidごとにGPSデータを分割中...", total=len(unique_ids), objectid="N/A")
            for id_val in unique_ids:
                # id_valが数値の場合もあるので、文字列に変換
                id_str = str(id_val)
                if id_str == "nan":
                    progress.advance(task)
                    continue
                # 該当IDの行のみ抽出
                sub_df = df_exploded[df_exploded["objectid_list"] == id_val].copy()
                
                # 補助カラム(objectid_list)は削除
                sub_df.drop(columns=["objectid_list"], inplace=True)
                
                # objectid列の値を該当IDで上書き（任意）
                sub_df["objectid"] = id_str
                
                # 日時列（date_time）をdatetime型に変換（エラーはNaTに）
                # sub_df["date_time"] = pd.to_datetime(sub_df["date_time"], errors="coerce")
                sub_df["date_time"] = pd.to_datetime(sub_df["date_time"])
                
                # 日時順にソートする
                sub_df.sort_values("date_time", inplace=True)
                
                # 連続しているレコードかどうかの判定：
                # 前後30秒以上の差分があれば、新たな走行区間とみなす
                sub_df['time_diff'] = sub_df["date_time"].diff()
                sub_df['segment'] = (sub_df['time_diff'] > pd.Timedelta(seconds=30)).cumsum()
                
                # objectid毎のサブディレクトリを作成（ここで文字列のid_strを利用）
                objectid_dir = os.path.join(self.csv_save_dir, id_str)
                os.makedirs(objectid_dir, exist_ok=True)
                
                # 各区間ごとに別ファイルとして保存
                for seg, df_seg in sub_df.groupby("segment"):
                    # 区間の最初のレコードの日時を取得
                    start_time = df_seg["date_time"].iloc[0]
                    # 日時を "YYYY-MM-DD_HH-MM-SS" 形式にフォーマット
                    dt_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
                    
                    # ファイル名は日時部分のみ（objectidはフォルダ名に含まれるため）
                    filename = f"{dt_str}.csv"
                    
                    # 補助カラムの削除
                    df_seg = df_seg.drop(columns=["time_diff", "segment"])
                    
                    # CSVとして保存（objectidのサブディレクトリに出力）
                    save_path = os.path.join(objectid_dir, filename)
                    df_seg.to_csv(save_path, index=False)
                progress.advance(task)
            progress.update(task, description="[green]objectidごとにGPSデータを分割中...完了")
            
        return self.csv_save_dir