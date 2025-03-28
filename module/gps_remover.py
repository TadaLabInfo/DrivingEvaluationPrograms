import os
import pandas as pd
import numpy as np
import itertools

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

class GPSRemover:
    """
    クラス: GPSRemover
    概要:
        このクラスは、GPSデータに含まれるobjectid情報の重複を処理するための機能を提供します。
        CSVファイルまたはDataFrameとしてGPSデータを読み込み、各レコードのobjectidをリスト形式に変換した上で、
        時刻情報に基づいて安定ソートを行い、連続するグループごとに特定のobjectid（候補ID）の重複を除去します。
        最終的には、補助カラムを削除し、修正したデータをCSVに保存またはDataFrameとして返します。
        中川追記: 簡単に言うと、何らかの本線を走っているとき、その付近の側道や別の道路の情報が混ざっている場合、それを取り除く処理を行うものです。
    メソッド:
        __init__(self, dir_path, gps_path):
            初期化メソッド。GPSデータの保存先ディレクトリと、対象のGPSデータCSVファイルまたはDataFrameを受け取ります。
            引数:
                dir_path (str):
                    GPSデータを保存するディレクトリのベースパス。
                gps_path (str または pandas.DataFrame):
                    GPSデータのCSVファイルパス、もしくは既に読み込まれたDataFrame。
        __call__(self):
            インスタンスを呼び出した際に実行されるメソッドです。
            主な処理:
                - objectid列の空欄処理、文字列変換およびリスト変換
                - date_time列に基づく安定ソート（mergesortを利用）
                - 全objectidの一意な候補IDを抽出し、各候補ごとに重複記録処理（process_candidateメソッドの呼び出し）
                - objectid_listをコロン区切りの文字列に戻し、補助カラムを削除
                - 処理後のデータをCSVファイルに保存し、DataFrameを返す
            戻り値:
                修正済みのpandas.DataFrame
        get_contiguous_groups(self, idx_array):
            ソート済みのインデックス配列またはリストから、連続するインデックスのグループを抽出します。
            引数:
                idx_array (list または numpy.array):
                    連続性の検出対象となる、ソート済みのインデックス集合
            戻り値:
                連続するインデックスのグループ（リストのリスト）
            処理概要:
                idx_array内で隣接する数字の差分が1でない部分を検出し、
                その分割点を基に連続するグループに分けて返す。
        process_candidate(self, df, candidate):
            指定された候補ID（candidate）に対して、データフレーム上で連続するレコード群ごとに以下の処理を行います。
            - グループ内に「candidateのみ」で構成されたレコードが存在するかを確認
            - 純粋なレコードが存在しない場合、複合記録から対象のcandidateを除外
            引数:
                df (pandas.DataFrame):
                    objectid_list列を含むデータフレーム（objectid_listは各レコードでIDのリスト）
                candidate (str):
                    対象となるobjectid（例："2888880"）
            戻り値:
                指定の候補IDに対し重複処理を実施した後のpandas.DataFrame
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
        """objectid列の重複を除去し、CSVに保存する
        """        
        
        # 空欄対策
        self.gps_df["objectid"] = self.gps_df["objectid"].fillna("").astype(str)
        
        # objectid列からスペース除去・リスト変換
        self.gps_df["objectid_list"] = self.gps_df["objectid"].apply(lambda s: [x.strip() for x in s.split(":") if x.strip()] if s.strip() != "" else [])
        
        # date_timeで安定ソート（同一日時の順序が変わらないよう mergesort を利用）
        self.gps_df = self.gps_df.sort_values("date_time", kind="mergesort").reset_index(drop=True)
        
        # 全レコードのobjectid_listから一意のIDを抽出
        all_ids = set(itertools.chain.from_iterable(self.gps_df["objectid_list"]))
        
        # 各IDについて重複処理を実施
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("objectidごとにGPSデータの重複を除去中...", total=len(all_ids))
            for candidate in all_ids:
                self.gps_df = self.process_candidate(self.gps_df, candidate)
                progress.advance(task)
            progress.update(task, description="[green]objectidごとにGPSデータの重複を除去中...完了")
        
        # objectid_listをcolon区切りの文字列に戻す
        self.gps_df["objectid"] = self.gps_df["objectid_list"].apply(lambda lst: ":".join(lst) if lst else "")
        
        # 補助カラム削除し保存
        self.gps_df.drop(columns=["objectid_list"], inplace=True)
        
        csv_out = os.path.join(self.dir_path, "gps_data_with_objectid_remove_duplicate.csv")
        self.gps_df.to_csv(csv_out, index=False)
        return self.gps_df
        
    def get_contiguous_groups(self, idx_array):
        """
        idx_array: 1次元のソート済みnumpy配列またはlist
        連続したindexのグループ（リストのリスト）を返す
        """
        groups = []
        if len(idx_array) == 0:
            return groups
        arr = np.array(idx_array)
        splits = np.where(np.diff(arr) != 1)[0]
        start = 0
        for split in splits:
            groups.append(arr[start:split+1].tolist())
            start = split + 1
        groups.append(arr[start:].tolist())
        return groups

    def process_candidate(self, df, candidate):
        """
        df: objectid_list列を持つDataFrame（objectid_listは[ str, ... ]）
        candidate: 対象交差点ID（例 "2888880"）
        
        df内でcandidateを含む行について、連続グループ内で
        「candidateのみであるレコード」が存在しなければ、グループ内の複合記録から対象IDを削除する。
        """
        mask = df["objectid_list"].apply(lambda lst: candidate in lst)
        if mask.sum() == 0:
            return df
        idxs = df[mask].index.tolist()
        idxs.sort()
        groups = self.get_contiguous_groups(idxs)
        for group in groups:
            # グループ内の各レコードについて、単独のcandidateレコードがあるか確認
            has_pure = any((len(df.at[i, "objectid_list"]) == 1 and df.at[i, "objectid_list"][0] == candidate) for i in group)
            if not has_pure:
                # 純粋なレコードがなければ、グループ内の複合記録から対象IDを除外
                for i in group:
                    lst = df.at[i, "objectid_list"]
                    if len(lst) > 1 and candidate in lst:
                        new_lst = [x for x in lst if x != candidate]
                        df.at[i, "objectid_list"] = new_lst
        return df