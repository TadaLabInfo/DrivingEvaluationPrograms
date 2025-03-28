from module.info_extractor import InfoExtractor
from module.crossroad_finder_vectorized import CrossRoadFinderVectorized
from module.gps_remover import GPSRemover
from module.gps_splitter import GPSSplitter
from module.turn_detection import TurnDetector
from module.create_kml import CreateKML
from module.create_movie_cv2 import CreateMovie

import logging
import inspect
import coloredlogs

# ログ設定：DEBUGレベル以上のメッセージを表示
logger = logging.getLogger("app")
coloredlogs.install(level="DEBUG")

# PIL.PngImagePluginとmatplotlib.font_managerとmatplotlib.pyplotの警告を非表示にする
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.pyplot").setLevel(logging.ERROR)

# 処理するディレクトリのパス
# dir_path = "ANJS001"
# dir_path = "ANJS001_temp2"
# dir_path = "ANJO001_temp"
dir_path = "ANJO001"

# 道路ネットワークデータのパス
# road_network_csv = "aichi-network4.csv"
road_network_csv = "test_network4.csv"

# 情報レベルのログで出力
logger.info(f"対象ディレクトリ: {dir_path}, 道路ネットワークデータ: {road_network_csv}")
# logger.debug(f"GPSデータ保存先: {gps_save_dir}")

logger.info("動画からGPSデータと顔向き角度の抽出を開始")
logger.debug(f"使用プログラム: {inspect.getfile(InfoExtractor)}")
# InfoExtractorのインスタンスを作成
info_extractor = InfoExtractor(
    dir_path=dir_path,  # 動画ファイルが格納されたディレクトリのパス
    info_save_dir=None,  # GPSデータの保存先ディレクトリのパス(指定なしだとdir_pathに"_info"が追記されたディレクトリ)
    use_cuda=True,  # GPUを利用する場合はTrue
    parallel=True,  # 並列処理を行う場合はTrue
    save_angle_video=True,  # 顔向き角度の動画を保存する場合はTrue
    save_angle_csv=True,  # 顔向き角度のCSVを保存する場合はTrue
    sampling_rate=15,  # GPSデータのサンプリングレート
    face_range=[
        0,
        30,
    ],  # 顔向き角度の可視範囲(check_face_range.pyで範囲は事前確認のこと)
)
# GPSデータ(重複削除&補完したもの)と顔向き角度を抽出
combine_info_df = info_extractor()
# # GPSデータ(重複削除&補完したもの)と顔向き角度のパス(ファイルを指定する場合)
# combine_info_df = "ANJO001_temp_info/combined_interpolated_fixed_gps_data.csv"

# 交差点情報の保存先ディレクトリを取得
info_save_dir = info_extractor.info_save_dir
# # 交差点情報の保存先ディレクトリ(ファイルを指定する場合)
# info_save_dir = "ANJO001_temp_info"

logger.info("道路ネットワークデータより交差点の検出を開始")
logger.debug(f"使用プログラム: {inspect.getfile(CrossRoadFinderVectorized)}")
# CrossRoadFinderVectorizedのインスタンスを作成
finder = CrossRoadFinderVectorized(
    dir_path=dir_path,  # 動画ファイルが格納されたディレクトリのパス
    road_network_path=road_network_csv,  # 道路ネットワークデータのパス
    info_path=combine_info_df,  # GPSデータのパス
    threshold=50,  # 交差点中心からの距離の閾値
    extension_distance=150,  # 交差点の拡張距離(thresholdが50、これが150なら200m分のGPSデータをCSVに含める)
    info_save_dir=info_save_dir,  # 交差点情報の保存先ディレクトリのパス
)
# 交差点IDを付与したGPSデータを取得
gps_with_objectid = finder()
# # 交差点ID付きのGPSデータのパス(ファイルを指定する場合)
# gps_with_objectid = "ANJO001_temp_info/gps_data_with_objectid.csv"

logger.info("objectidごとにGPSデータの重複を除去開始")
logger.debug(f"使用プログラム: {inspect.getfile(GPSRemover)}")
# GPS_Removerのインスタンスを作成
remover = GPSRemover(
    dir_path=info_save_dir,  # GPSデータの保存先ディレクトリのパス
    gps_path=gps_with_objectid,  # 交差点ID付きのGPSデータのパス
)
# objectid列の重複を除去し、CSVに保存
gps_with_objectid_removed = remover()
# # 交差点ID付き(重複削除済み)のGPSデータのパス(ファイルを指定する場合)
# gps_with_objectid_removed = "ANJO001_temp_info/gps_data_with_objectid_remove_duplicate.csv"  

logger.info("objectidごとにGPSデータを分割開始")
logger.debug(f"使用プログラム: {inspect.getfile(GPSSplitter)}")
# GPSSplitterのインスタンスを作成
splitter = GPSSplitter(
    dir_path=info_save_dir,  # GPSデータの保存先ディレクトリのパス
    gps_path=gps_with_objectid_removed,  # 交差点ID付き(重複削除済み)のGPSデータのパス
)
# objectidごとに分割したGPSデータを保存
splitted_save_dir = splitter()
# # # objectidごとに分割したGPSデータの保存先ディレクトリのパス(ファイルを指定する場合)
# splitted_save_dir = "ANJO001_info/objectid_csvs"

logger.info("直進(STR)・右折(RGT)・左折(LFT)を判定開始")
logger.debug(f"使用プログラム: {inspect.getfile(TurnDetector)}")
# TurnDetectorのインスタンスを作成
turn_detector = TurnDetector(
    dir_path=splitted_save_dir,  # GPSデータの保存先ディレクトリのパス
    net_path=road_network_csv,  # 道路ネットワークデータのパス
    distance_thresh=50,  # 交差点中心からの距離の閾値
)
# 右左折を判定
turn_detector()

logger.info("kmlファイルの作成を開始")
# CreateKMLのインスタンスを作成
kml_creator = CreateKML(
    dir_path=splitted_save_dir,  # GPSデータの保存先ディレクトリのパス
    road_network_csv=road_network_csv,  # 道路ネットワークデータのパス
)
# KMLファイルを作成
kml_creator()

logger.info("動画の作成を開始")
# CreateMovieのインスタンスを作成
movie_creator = CreateMovie(
    dir_path=splitted_save_dir,  # GPSデータの保存先ディレクトリのパス
    network_csv=road_network_csv,  # 道路ネットワークデータのパス
    accum_distance=50,  # 50mごとにフレームを追加
)
movie_creator()

logger.info("処理が完了しました。")
