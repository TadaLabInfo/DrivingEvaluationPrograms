#!/usr/bin/env python3
import subprocess
import struct
import os
import sys
import csv
import datetime
import re

def extract_video_datetime(filename):
    """ビデオファイル名から日時情報を抽出する"""
    # "NN3_250422-170742-000060_1.MP4" のようなファイル名から日時を抽出
    # 250422-170742 は 2022/04/25 17:07:42 と解釈する
    pattern = r'.*_(\d{6})-(\d{6}).*\.MP4'
    match = re.search(pattern, filename, re.IGNORECASE)
    
    if match:
        date_part = match.group(1)  # 例: 250422
        time_part = match.group(2)  # 例: 170742
        
        try:
            day = int(date_part[0:2])
            month = int(date_part[2:4])
            year = 2000 + int(date_part[4:6])  # 22 -> 2022
            
            hour = int(time_part[0:2])
            minute = int(time_part[2:4])
            second = int(time_part[4:6])
            
            return datetime.datetime(year, month, day, hour, minute, second)
        except ValueError:
            return None
    
    return None

def extract_gps_data(video_file, output_csv="gps_data_extracted.csv"):
    """
    ドライブレコーダー動画からGPSデータを抽出する完全なプロセスを1つのPythonスクリプトで実行
    
    Args:
        video_file: ドライブレコーダー動画ファイルのパス
        output_csv: 抽出したGPSデータを保存するCSVファイルのパス
    """
    print(f"Processing video file: {video_file}")
    
    # ビデオファイル名から基準となる日時を抽出
    video_datetime = extract_video_datetime(os.path.basename(video_file))
    if video_datetime:
        print(f"Extracted date/time from filename: {video_datetime}")
    else:
        print("Could not extract date/time from filename")
    
    # 作業用ディレクトリを作成
    tmp_dir = "tmp_metadata"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # メタデータストリームの抽出（ffmpegを使用）
    metadata_bin = os.path.join(tmp_dir, "metadata_stream.bin")
    
    # ffmpeg コマンドを実行してメタデータを抽出
    cmd = [
        "ffmpeg", "-i", video_file, 
        "-map", "0:2", "-c:d", "copy", 
        "-f", "data", metadata_bin
    ]
    
    try:
        print("Extracting metadata stream using ffmpeg...")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Metadata extracted to {metadata_bin}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing ffmpeg: {e}")
        # 2番目のメタデータストリームを試す
        cmd[3] = "0:3"
        try:
            print("Trying alternate metadata stream...")
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Metadata extracted to {metadata_bin}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting metadata: {e}")
            return False
    
    # GPSデータの解析と変換
    if not os.path.exists(metadata_bin) or os.path.getsize(metadata_bin) == 0:
        print("Error: Metadata extraction failed or empty file")
        return False
    
    # メタデータを解析してGPS座標に変換
    process_metadata(metadata_bin, output_csv, video_datetime)
    
    print(f"GPS data extracted to {output_csv}")
    return True

def process_metadata(metadata_file, output_csv, base_datetime=None):
    """メタデータバイナリファイルを解析してGPS座標を抽出"""
    with open(metadata_file, 'rb') as f:
        data = f.read()
    
    print(f"Analyzing metadata - {len(data)} bytes")
    
    # 基準となるレコード番号とタイムスタンプ値を保存するための変数
    base_record_num = 0
    base_timestamp = None
    
    # CSV出力ファイルを作成
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Record', 'Timestamp', 'DateTime', 'Latitude', 'Longitude', 'Extra_Bytes'])
        
        records = []
        
        # 32バイトごとにレコードを処理し、まずデータを収集
        for i in range(0, len(data), 32):
            if i + 32 > len(data):
                break
            
            record = data[i:i+32]
            
            # 最初の16バイトをNMEA形式の緯度・経度と解釈
            try:
                values = struct.unpack("<dd", record[:16])
                
                # NMEA値（DDMM.MMMM形式）
                nmea_lat = values[0]
                nmea_lon = values[1]
                
                # DDMM.MMMMから10進法の度数に変換
                # 緯度: 最初の2桁が度数、残りが10進法の分
                lat_deg = int(nmea_lat / 100)
                lat_min = nmea_lat - (lat_deg * 100)
                latitude = lat_deg + (lat_min / 60)
                
                # 経度: 最初の3桁が度数、残りが10進法の分
                lon_deg = int(nmea_lon / 100)
                lon_min = nmea_lon - (lon_deg * 100)
                longitude = lon_deg + (lon_min / 60)
                
                # 追加バイトにはタイムスタンプやメタデータが含まれている可能性あり
                extra_bytes = " ".join([f"{b:02x}" for b in record[16:24]])
                extra_data = struct.unpack("<Q", record[16:24])[0]
                
                # 最初の有効なレコードのタイムスタンプを保存
                if base_timestamp is None:
                    base_timestamp = extra_data
                    base_record_num = i//32
                
                records.append({
                    'record_num': i//32,
                    'timestamp': extra_data,
                    'latitude': latitude,
                    'longitude': longitude,
                    'extra_bytes': extra_bytes
                })
                
            except Exception as e:
                print(f"Error processing record {i//32}: {e}")
        
        # タイムスタンプ間の差分から日時を計算
        for record in records:
            record_num = record['record_num']
            timestamp = record['timestamp']
            
            date_time_str = "Unknown"
            
            # ファイル名から抽出した日時がある場合は、それを基準に日時を計算
            if base_datetime:
                # レコード番号の差から秒数を計算
                # 仮定：GPSデータは約1秒間隔で記録されている
                seconds_diff = (record_num - base_record_num) * 1.0
                
                # 全体で約1分間になるように調整
                total_records = len(records)
                if total_records > 1:
                    # 約1分（60秒）をレコード数で割って、1レコードあたりの秒数を計算
                    seconds_per_record = 60.0 / (total_records - 1)
                    seconds_diff = (record_num - base_record_num) * seconds_per_record
                
                # 基準日時に秒数を加算
                date_time = base_datetime + datetime.timedelta(seconds=seconds_diff)
                date_time_str = date_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # ミリ秒まで表示
            
            # CSVに出力
            csvwriter.writerow([
                record_num,
                timestamp,
                date_time_str,
                record['latitude'],
                record['longitude'],
                record['extra_bytes']
            ])
    
    print(f"Extracted GPS data saved to {output_csv}")

def main():
    """メイン実行関数"""
    if len(sys.argv) < 2:
        print("Usage: python extract_gps_all_in_one.py <mp4_file_path>")
        sys.exit(1)
    
    video_file = sys.argv[1]
    output_csv = "gps_data_extracted.csv"
    
    if not os.path.exists(video_file):
        print(f"Error: File {video_file} does not exist")
        sys.exit(1)
    
    # GPSデータを抽出
    if extract_gps_data(video_file, output_csv):
        print(f"\nExtraction complete!")
        print(f"GPS data saved to: {output_csv}")
        
        # 一時フォルダとファイルを削除
        tmp_dir = "tmp_metadata"
        if os.path.exists(tmp_dir):
            try:
                metadata_bin = os.path.join(tmp_dir, "metadata_stream.bin")
                if os.path.exists(metadata_bin):
                    os.remove(metadata_bin)
                os.rmdir(tmp_dir)
                print(f"Temporary files and directory '{tmp_dir}' removed")
            except Exception as e:
                print(f"Warning: Could not remove temporary files: {e}")
    else:
        print("GPS data extraction failed")
        sys.exit(1)

if __name__ == "__main__":
    main()