import sys
import os
import re
import subprocess

def get_package_location(package_name):
    """
    sys.executable を用いて pip show を実行し、
    出力から 'Location' 行を抽出してパッケージのインストール先ディレクトリを返す。
    """
    try:
        # venv環境下であることを前提に、sys.executableでpipモジュールを実行
        pip_cmd = [sys.executable, "-m", "pip", "show", package_name]
        output = subprocess.check_output(pip_cmd, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: pip show {package_name} に失敗しました。 {e}")
        return None

    # 出力から "Location:" 行を正規表現で抽出
    m = re.search(r"Location:\s*(.+)", output)
    if m:
        location = m.group(1).strip()
        print(f"パッケージ '{package_name}' のインストール先: {location}")
        return location
    else:
        print("Error: pip show の出力から Location を抽出できませんでした。")
        return None

def patch_file(file_path):
    """
    指定のファイル内の 'np.int'（ただし np.int64 は除外対象）を 'np.int64' に置換し、
    元のファイルはバックアップとして同じディレクトリに拡張子 .bak で保存する。
    """
    if not os.path.exists(file_path):
        print(f"Error: 指定ファイルが見つかりません: {file_path}")
        return

    # ファイル内容の読み込み
    with open(file_path, 'r', encoding='utf-8') as f:
        original_text = f.read()

    # バックアップ作成
    backup_path = file_path + '.bak'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_text)
    print(f"バックアップを作成しました: {backup_path}")

    # 「np.int」だが「np.int64」は置換しない正規表現パターン
    patched_text = re.sub(r'np\.int(?!64)', "np.int64", original_text)

    # 上書き保存
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(patched_text)
    print(f"ファイルをパッチしました: {file_path}")

def main():
    package_name = "insightface"
    package_location = get_package_location(package_name)
    if package_location is None:
        return

    # insightface/app/face_analysis.py のパスを組み立てる
    target_file = os.path.join(package_location, "insightface", "app", "face_analysis.py")
    print(f"パッチ対象ファイル: {target_file}")

    patch_file(target_file)

if __name__ == '__main__':
    main()