import matplotlib
from onnx import save
from .insightface_calculator import InsightFaceCalculator
from .sixdrepnet_calculator import SixDRepNetCalculator
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


class FaceAngleHandler:
    def __init__(
        self,
        use_cuda=False,
        parallel=False,
        save_angle_video=False,
        save_angle_csv=False,
        save_dir="info",
        face_range=[0, 100],
        replace_smooth=True,
        smooth=True,
    ):
        """動画から顔の角度を取得するクラス

        Args:
            use_cuda (bool, optional): GPUを利用するかどうか. Defaults to False.
            parallel (bool, optional): InsightFaceと6DRepNetの処理を並列実行するかどうか. Defaults to False.
        """
        self.parallel = parallel
        # InsightFaceCalculatorのインスタンスを作成
        self.insight_ins = InsightFaceCalculator(
            use_cuda=use_cuda, save_angle_video=save_angle_video, save_dir=save_dir
        )
        # SixDRepNetCalculatorのインスタンスを作成
        self.sixd_ins = SixDRepNetCalculator(
            use_cuda=use_cuda, save_angle_video=save_angle_video, save_dir=save_dir
        )

        self.save_dir = save_dir
        self.save_angle_csv = save_angle_csv

        self.face_range = face_range
        
        self.replace_smooth = replace_smooth
        self.smooth = smooth
        
        # insightと6drepnetの保存先ディレクトリで、ない場合は作成
        self.save_dir_insight = os.path.join(self.save_dir, "insight")
        self.save_dir_repnet = os.path.join(self.save_dir, "6drepnet")
        os.makedirs(self.save_dir_insight, exist_ok=True)
        os.makedirs(self.save_dir_repnet, exist_ok=True)

    def __call__(self, video_path, progress=None):
        """動画から顔の角度を取得する（Noneが含まれている場合は後で補完処理などを実施）

        Args:
            video_path (str): 動画のパス
            face_range (list, optional): 顔のフレーム範囲例えば左30%のみを取得するなら[0, 30]等. Defaults to [0, 100].

        Returns:
            tuple: InsightFaceのyaw角度, InsightFaceの重畳画像, SixDRepNetのyaw角度
        """
        self.video_path = video_path

        if self.parallel:
            # 並列処理：ThreadPoolExecutorによりInsightFaceと6DRepNetの計算を並列に実施
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_insight = executor.submit(
                    self.insight_ins, self.video_path, self.face_range, progress
                )
                future_repnet = executor.submit(
                    self.sixd_ins, self.video_path, self.face_range, progress
                )
                # 両方完了するまで待機し結果を取得
                insight_yaw, insight_det = future_insight.result()
                repnet_yaw = future_repnet.result()
        else:
            # シーケンシャル実行
            # insight_yaw, insight_det = self.insight_ins(self.video_path, face_range, progress=progress)
            insight_yaw, insight_det = self.insight_ins(
                self.video_path, self.face_range, progress=progress
            )
            repnet_yaw = self.sixd_ins(
                self.video_path, self.face_range, progress=progress
            )

        # Noneがある場合は補完
        insight_yaw = self.interpolate_yaw(insight_yaw)
        repnet_yaw = self.interpolate_yaw(repnet_yaw)

        if self.save_angle_csv:
            # insight_yawとinsight_detを結合
            insight_data = np.column_stack((insight_yaw, insight_det))

            # 動画ファイル名
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]

            # insightと6drepnetのcsvファイルパス
            csv_path_insight = os.path.join(self.save_dir_insight, video_name + ".csv")
            csv_path_repnet = os.path.join(self.save_dir_repnet, video_name + ".csv")

            # insightと6drepnetのcsvファイルに保存
            np.savetxt(
                csv_path_insight,  # insightのcsvファイルパス
                insight_data,  # insight_yawとinsight_detを結合したデータ
                delimiter=",",  # 区切り文字
                header="insight_yaw,insight_det",  # ヘッダー
                comments="",  # コメント
            )
            np.savetxt(
                csv_path_repnet,  # 6drepnetのcsvファイルパス
                repnet_yaw,  # 6drepnetのyaw角度
                delimiter=",",  # 区切り文字
                header="repnet_yaw",  # ヘッダー
                comments="",  # コメント
            )

        return insight_yaw, insight_det, repnet_yaw

    def fix(
        self,
        insight_yaw,
        insight_det,
        repnet_yaw,
        insight_median,
        repnet_median,
        gps_df_interpolate,
        y_min=-180,
        y_max=180,
        save_path="yaw_plot.png",
    ):
        # InsightFaceとSixDRepNetのYaw角度を中央値を基準に修正
        insight_yaw = self.calibration_yaw(insight_yaw, median=insight_median)
        repnet_yaw = self.calibration_yaw(repnet_yaw, median=repnet_median)
        
        # 修正前のinsight_yawの値を保持しておく（後で置換されたかを判定するため）
        original_insight_yaw = np.array(insight_yaw).copy()
        
        # InsightFaceのYaw角度抜けをSixDRepNetのYaw角度で補完
        if self.replace_smooth:
            fixed_yaw = self.replace_yaw_smooth(insight_yaw, insight_det, repnet_yaw)
        else:
            fixed_yaw = self.replace_yaw(insight_yaw, insight_det, repnet_yaw)
            
        # スムージング処理
        if self.smooth:
            fixed_yaw = self.smooth_yaw(fixed_yaw, cutoff_hz=5)
            
        # 整数値に丸める
        fixed_yaw = np.around(fixed_yaw).astype(int)
        
        # 生成したfixed_yawと元のinsight_yawとの差分で、SixDRepNetで置き換えた箇所のマスクを作成（1:置換、0:そのまま）
        rep_mask = (fixed_yaw != np.around(original_insight_yaw).astype(int)).astype(int)
        
        # GPSデータに対して、最も近い時刻のface_angleを追加する
        fixed_yaw_df = self.sync_face_angle_to_gps(fixed_yaw, gps_df_interpolate)
        
        # 上記と同様に、rep_maskもGPSデータに同期させる
        start_time = gps_df_interpolate["date_time"].iloc[0]
        end_time = gps_df_interpolate["date_time"].iloc[-1]
        rep_time = np.linspace(start_time.value, end_time.value, len(rep_mask))
        rep_time = pd.to_datetime(rep_time)
        # rep_time = rep_time.tz_localize("UTC").tz_convert("Asia/Tokyo")
        rep_df = pd.DataFrame({"6dreplaced": rep_mask, "date_time": rep_time})
        nearest_indices = np.searchsorted(rep_df["date_time"], fixed_yaw_df["date_time"])
        fixed_yaw_df["6dreplaced"] = rep_df["6dreplaced"].iloc[nearest_indices].values

        save_file_name = os.path.splitext(os.path.basename(save_path))[0] + "_plot.png"
        save_path = os.path.join(self.save_dir, "insight_6drepnet", save_file_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # self.plot_yaw(fixed_yaw, y_min=y_min, y_max=y_max, save_path=save_path)
        self.plot_yaw(fixed_yaw, y_min=y_min, y_max=y_max, save_path=save_path, replaced_mask=fixed_yaw_df["6dreplaced"])

        return fixed_yaw_df

    def calc_yaw_median(self, yaw_list):
        """Yaw角度の中央値を計算する

        Args:
            yaw_list (list): Yaw角度

        Returns:
            float: Yaw角度の中央値
        """
        yaw_stack = np.hstack(yaw_list)
        median_yaw = np.median(yaw_stack)
        return median_yaw

    def calibration_yaw(self, yaw_list, median=None):
        """yaw角度から中央値(正面を向いている)を計算し、その差分で修正する

        Args:
            yaw_list (list): yaw角度
            median (float, optional): 中央値. Defaults to None

        Returns:
            list: 修正後のyaw角度
        """
        if median is None:
            median_yaw = self.calc_yaw_median(yaw_list)
        else:
            median_yaw = median
        fixed_yaw = yaw_list - median_yaw

        return fixed_yaw

    def replace_yaw(self, insight_yaw, insight_det, repnet_yaw, det_threshold=0.725):
        """InsightFaceのYaw角度抜けをSixDRepNetのYaw角度で補完する
        具体的には、insight_detがdet_threshold以下の場合、そのレコードのinsight_yawをrepnet_yawで置き換える
        replace_yaw_smoothからスムージング処理を削除したもの

        Args:
            insight_yaw (list): InsightFaceのYaw角度
            insight_det (list): InsightFaceのdet_score
            repnet_yaw (list): SixDRepNetのYaw角度

        Returns:
            list: 補完後のYaw角度
        """
        replace_idx = np.where(np.array(insight_det) <= det_threshold)[0]
        print(f"補完対象のインデックス: {replace_idx}")
        for idx in replace_idx:
            insight_yaw[idx] = repnet_yaw[idx]

        return insight_yaw

    def replace_yaw_smooth(self, insight_yaw, insight_det, repnet_yaw, det_threshold=0.725):
        """
        置換および平滑化を行う関数。
        引数:
            insight_yaw (list または numpy配列): 元のヨー角（左右の回転角度）を表す値のリストまたは配列。
            insight_det (list または numpy配列): 各フレームの検出信頼度。値が低い場合、repnet_yawによる置換対象とする。
            repnet_yaw (list または numpy配列): repnetモデルから得られたヨー角の値のリストまたは配列。
        処理内容:
            1. insight_detの値がdet_threshold以下の場合、対応するinsight_yawの値をrepnet_yawの値で置換する。
            2. 置換されたインデックスの連続区間をグループ化する。
            3. 各グループの開始および終了位置の周囲（最大±5フレーム）について、近傍ウィンドウの平均値で平滑化を行い、そのウィンドウ内の値を平均値で更新する。
        返り値:
            list: 置換および平滑化後のヨー角の値を保持するリスト。
        備考:
            - 関数内部でnumpyをインポートして処理を行っているため、入力データはnumpy配列またはリスト形式であることが望ましい。
            - 置換および平滑化の処理は、境界条件に注意しながら安全に行われる。
        """
        # 入力のinsight_yawをコピーして、元データを保持しつつ置換を行う
        replaced_yaw = insight_yaw[:]  
        
        # 信頼度がdet_threshold以下の箇所をTrueとするマスクを作成する
        replaced_mask = np.array(insight_det) <= det_threshold
        # 置換対象となるインデックスを抽出
        replace_idx = np.where(replaced_mask)[0]
        
        # 各インデックスに対してrepnet_yawの値で置換
        for idx in replace_idx:
            replaced_yaw[idx] = repnet_yaw[idx]
        
        # 置換された連続区間をグループ化するためのリスト
        groups = []
        if len(replace_idx) > 0:
            # 最初のインデックスから初期グループを作成
            current_group = [replace_idx[0]]
            # 以降のインデックスについて検証
            for idx in replace_idx[1:]:
                # 連続したインデックスならcurrent_groupに追加
                if idx == current_group[-1] + 1:
                    current_group.append(idx)
                else:
                    # 連続していなければ、現在のグループをgroupsに追加し、新たなグループを開始
                    groups.append(current_group)
                    current_group = [idx]
            # 最後のグループを追加
            groups.append(current_group)
        
        # 置換後のyawデータをnumpy配列に変換（浮動小数点数型）
        smoothed_yaw = np.array(replaced_yaw, dtype=float)
        n = len(smoothed_yaw)
        
        # 各置換グループについて、開始位置と終了位置の周辺で平滑化を実施
        for group in groups:
            s = group[0]  # グループ開始インデックス
            e = group[-1]  # グループ終了インデックス

            # グループ開始位置の前に置換されていない信頼できる値がある場合、平滑化
            if s > 0 and (s - 1 >= 0) and (not replaced_mask[s - 1]):
                # 平滑化ウィンドウの開始位置は、sから5フレーム前（範囲を超えないように）
                win_start = max(0, s - 5)
                # 終了位置は、sから5フレーム後（境界を超えないように）
                win_end = min(n, s + 5 + 1)
                # ウィンドウ内の値の平均値を算出
                window = smoothed_yaw[win_start:win_end]
                avg = window.mean()
                # ウィンドウ内の全値を平均値で更新
                smoothed_yaw[win_start:win_end] = avg

            # グループ終了位置の後に置換されていない信頼できる値がある場合、平滑化
            if e < n - 1 and (e + 1 < n) and (not replaced_mask[e + 1]):
                # 平滑化ウィンドウの開始位置は、eから5フレーム前（範囲を超えないように）
                win_start = max(0, e - 5)
                # 終了位置は、eから5フレーム後（範囲を超えないように）
                win_end = min(n, e + 5 + 1)
                # ウィンドウ内の値の平均値を算出
                window = smoothed_yaw[win_start:win_end]
                avg = window.mean()
                # ウィンドウ内の全値を平均値で更新
                smoothed_yaw[win_start:win_end] = avg

        # 平滑化処理後のyaw値をリストに変換して返却
        return smoothed_yaw.tolist()

    def interpolate_yaw(self, yaw_list):
        """
        このメソッドは、yaw値のリストに対して線形補間を行い、欠損値 (None) を補完します。
        パラメータ:
            yaw_list (list): yaw値のリスト。リスト内の各要素は数値またはNoneであり、Noneは欠損値として扱われます。
        戻り値:
            list: 補間されたyaw値のリスト。補間後の値は小数点以下を四捨五入して整数に変換されています。
        注意:
            - yaw_list内の全ての値が欠損値の場合、入力リストがそのまま返されます。
            - NumPyの関数 (np.interp, np.isnan, np.around など) を利用して線形補間と丸め処理を行っています。
        """
        # Noneの値をnp.nanに置き換え、数値はそのまま保持してnumpy配列に変換
        yaw_arr = np.array([np.nan if v is None else v for v in yaw_list], dtype=float)
        # 補間のためのインデックス配列を作成
        indices = np.arange(len(yaw_arr))
        # 有効な数値（np.nanでない値）のマスクを作成
        valid = ~np.isnan(yaw_arr)

        # 有効な数値が一つも存在しない場合は、元のリストをそのまま返す
        if np.sum(valid) == 0:
            return yaw_list

        # 有効な数値を利用して線形補間を実施
        interpolated = np.interp(indices, indices[valid], yaw_arr[valid])
        # 補間結果を四捨五入して整数に変換する
        rounded = np.around(interpolated).astype(int)
        # リスト形式に変換して返す
        return rounded.tolist()

    def smooth_yaw(self, yaw_list, cutoff_hz):
        """
        角度リストを滑らかにする関数
        この関数は、入力された角度リスト（yaw_list）に対してフーリエ変換を行い、
        指定されたカットオフ周波数（cutoff_hz）より高い周波数成分を除去することで、
        角度の変動を平滑化します。平滑化後、逆フーリエ変換により元の時間領域へ戻し、
        実数部分を整数に四捨五入して出力します。
        パラメータ:
            yaw_list (list[int]): 入力の角度リスト（yaw角、整数値）　　# 入力角度のリストです。
            cutoff_hz (float): カットオフ周波数（Hz単位）　　　　　　　# この周波数を超える成分が除去されます。
        戻り値:
            list[int]: 滑らかに補正された角度のリスト　　　　　　　　　# フィルタリング後の整数値の角度リスト。
        注意:
            - yaw_listが空の場合、そのまま空のリストを返します。
            - フーリエ変換と逆フーリエ変換を利用しているため、計算の精度や丸め処理によっては
              若干の誤差が生じる可能性があります。
        """
        # yaw_listの長さを取得
        N = len(yaw_list)
        # yaw_listが空の場合は、そのまま返す
        if N == 0:
            return yaw_list

        # フーリエ変換を実施し、周波数領域の係数を取得
        fft_coeff = np.fft.fft(yaw_list)
        # 各係数に対応する周波数を計算（動画のFPSに基づく）
        freq = np.fft.fftfreq(N, d=1 / float(self.__get_fps()))
        # 指定したカットオフ周波数より高い成分を除去するため、該当する係数を0に設定
        fft_coeff[np.abs(freq) > cutoff_hz] = 0
        # 逆フーリエ変換を行い、平滑化された信号を取得
        filtered = np.fft.ifft(fft_coeff)
        # 実数部を取り出し、四捨五入して整数に変換
        smoothed = np.around(np.real(filtered)).astype(int)
        # リスト形式に変換して返す
        return smoothed.tolist()

    def __get_fps(self):
        """
        このメソッドは、指定された動画ファイルのフレームレート (FPS) を取得します。
        cv2.VideoCapture を使用して動画ファイルを開き、CAP_PROP_FPS プロパティを参照してFPSを読み取ります。
        取得したFPSの値が0の場合、デフォルト値として30.0を返します。
        Returns:
            float: 動画ファイルのフレームレートまたは、FPSが0の場合は30.0
        """
        import cv2

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps == 0:
            fps = 30.0
        return fps

    def plot_yaw(self, yaw_list, y_min=None, y_max=None, save_path="yaw_plot.png", replaced_mask=None):
        """
        yaw_list内のyaw角データをフレームごとにプロットし、指定された範囲でy軸を調整してグラフを画像として保存する関数です。
        引数:
            yaw_list (listまたはnumpy.ndarray): 各フレームにおけるyaw角度の値のリスト。
            y_min (数値またはNone, オプション): y軸の下限値。Noneの場合、下限値は自動設定されます。
            y_max (数値またはNone, オプション): y軸の上限値。Noneの場合、上限値は自動設定されます。
            save_path (str, オプション): プロット画像を保存するファイルパス。既定値は"yaw_plot.png"です。
            replaced_mask (list または numpy.ndarray, オプション): 6DRepNetで置き換えられた箇所を示すマスク。
        戻り値:
            なし。プロットは指定されたパスにPNG形式の画像として保存されます。
        """
        # X軸のデータとしてフレーム番号を生成
        x = np.arange(len(yaw_list))
        # Y軸のデータとしてyaw角度のリストをnumpy配列に変換
        y = np.array(yaw_list)

        # プロットの全体サイズを設定して新しいフィギュアを作成
        plt.figure(figsize=(12, 6))

        # replaced_maskが指定されている場合、その値に基づいて線の色を変更
        if replaced_mask is not None:
            replaced_mask = np.array(replaced_mask)
            
            # replaced_maskの長さを調整
            if len(replaced_mask) > len(yaw_list):
                replaced_mask = replaced_mask[:len(yaw_list)]
            elif len(replaced_mask) < len(yaw_list):
                # 最後の値で埋める
                replaced_mask = np.pad(replaced_mask, (0, len(yaw_list) - len(replaced_mask)), 
                                    mode='constant', constant_values=replaced_mask[-1])
            
            # 線のセグメントを分けて描画
            current_color = 'green'  # デフォルトの色
            start_index = 0
            
            for i in range(1, len(x)):
                # 色が変わるか、最後の要素に到達した場合
                if replaced_mask[i] != replaced_mask[start_index] or i == len(x) - 1:
                    color = 'red' if replaced_mask[start_index] == 1 else 'green'
                    # 最後の要素の場合は、そのセグメントも描画
                    end_index = i + 1 if i == len(x) - 1 else i
                    plt.plot(x[start_index:end_index], y[start_index:end_index], 
                            marker="", linestyle="-", color=color)
                    start_index = i
        else:
            # replaced_maskが指定されていない場合は、従来通り青い線でプロット
            plt.plot(x, y, marker="", linestyle="-", color="blue", label="Yaw angle")

        # X軸に「Frame」というラベルを設定
        plt.xlabel("Frame")
        # Y軸に「Yaw angle (degrees)」というラベルを設定
        plt.ylabel("Yaw angle (degrees)")
        # プロットタイトルを設定
        plt.title("Yaw angle vs Frame")
        # グリッド線を表示して見やすくする
        plt.grid(True)
        # プロットの凡例を表示
        plt.legend()

        # y_minとy_maxが定義されている場合は、Y軸の範囲を固定する
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)

        # 指定されたパスにプロットを画像として保存（余白を調整）
        plt.savefig(save_path, bbox_inches="tight")
        # 使用したフィギュアを閉じ、リソースを解放
        plt.close()

    def sync_face_angle_to_gps(self, face_angles, gps_df):
        """
        同期処理により、顔の向き角度データとGPSデータを統合する関数。

        引数:
            face_angles (array-like):
                顔の向き角度の配列またはリスト。
            gps_df (pandas.DataFrame):
                GPSデータを含むデータフレーム。必ず 'date_time' 列が存在し、時系列の情報が格納されていること。

        処理概要:
            1. gps_dfから最初と最後の時刻（date_time）を取得する。
            2. 取得した時刻範囲内で、face_anglesの数に合わせた等間隔の時刻配列を生成する。
            3. 生成した時刻配列をUTCタイムゾーンにローカライズし、その後日本標準時（JST）に変換する。
            4. GPSデータの各時刻に対して、顔向き角度データの時刻で最も近いインデックスを検索する。
            5. 検索した結果を用いて、対応する顔向き角度をGPSデータに 'face_angle' 列として追加する。

        戻り値:
            pandas.DataFrame:
                'face_angle' 列が追加された同期済みのGPSデータフレーム。
        """
        # GPSデータの時刻の最初と最後を取得
        start_time = gps_df["date_time"].iloc[0]
        end_time = gps_df["date_time"].iloc[-1]
        # 顔向き角度用の時刻を作成
        face_angles_time = np.linspace(
            start_time.value, end_time.value, len(face_angles)
        )
        face_angles_time = pd.to_datetime(face_angles_time)
        face_angles_df = pd.DataFrame(
            {"face_angle": face_angles, "date_time": face_angles_time}
        )
        # # UTCタイムゾーンを設定
        # face_angles_df["date_time"] = face_angles_df["date_time"].dt.tz_localize("UTC")
        # # 日本標準時（JST）に変換
        # face_angles_df["date_time"] = face_angles_df["date_time"].dt.tz_convert(
        #     "Asia/Tokyo"
        # )
        # GPSデータの時刻に最も近い顔向き角度を取得
        nearest_indices = np.searchsorted(
            face_angles_df["date_time"], gps_df["date_time"]
        )
        # それをGPSデータに追加
        gps_df["face_angle"] = face_angles_df["face_angle"].iloc[nearest_indices].values
        return gps_df