import os
import re

# 探索するルートディレクトリ（適宜変更してください）
root_dir = r"ANJS001_1023_info\objectid_csvs"

# ファイル名の先頭部分が日付と時刻になっているものをマッチさせる正規表現
# 例: 2024-10-23_16-08-56_stop0_sig1_STR_link_6272377_to_6272377.csv
# から2024-10-23_16-08-56を抽出
pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_.*\.csv$", re.IGNORECASE)

for root, dirs, files in os.walk(root_dir):
    for filename in files:
        filepath = os.path.join(root, filename)
        lower_filename = filename.lower()
        # CSVファイルの場合
        if lower_filename.endswith(".csv"):
            m = pattern.match(filename)
            if m:
                new_filename = m.group(1) + ".csv"
                new_path = os.path.join(root, new_filename)
                # 新しい名前が既に存在する場合は上書きされるため、注意してください
                print("Renaming:", filepath, "->", new_path)
                os.rename(filepath, new_path)
        # KMLファイルの場合はファイル自体を削除する
        if lower_filename.endswith(".kml"):
            print("Deleting kml:", filepath)
            os.remove(filepath)
        if lower_filename.endswith(".mp4"):
            print("Deleting mp4:", filepath)
            os.remove(filepath)