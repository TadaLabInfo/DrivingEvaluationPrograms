import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(r"ANJO001\NNF_250310-142157-000038_1.AVI")
    if not cap.isOpened():
        print("エラー: ビデオストリームを開けませんでした。")
        return

    visible_range = [0, 30]
    visible_start, visible_end = visible_range
    alpha = 0.8  # 重畳画像の透過率
    fast_forward_frames = 10  # 早送り時にスキップするフレーム数

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        start_x = int(width * (visible_start / 100.0))
        end_x = int(width * (visible_end / 100.0))

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (start_x, height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (end_x, 0), (width, height), (0, 0, 0), -1)
        blended_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # 下半分のみを切り出して表示
        lower_half = blended_frame[height//2 : height, :]

        cv2.imshow("Video", lower_half)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        # "f" キーで早送り（フレームをスキップ）
        elif key == ord('f'):
            for _ in range(fast_forward_frames):
                cap.grab()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
