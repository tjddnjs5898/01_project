import cv2
import time
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("yolov8n.pt")

# 영상 파일 경로
video_path = r'../../img/car.mov'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상 열기 실패")
    exit()

# ROI 설정 (사각지대 영역 비율)
roi_ratio = {
    "x1": 0.75,
    "y1": 0.4,
    "x2": 0.98,
    "y2": 0.95
}

vehicle_times = {}      # 차량 시간 추적: key=center, value={'start_time':..., 'counted':...}
danger_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("영상 종료")
        break

    height, width, _ = frame.shape
    roi_top_left = (int(width * roi_ratio["x1"]), int(height * roi_ratio["y1"]))
    roi_bottom_right = (int(width * roi_ratio["x2"]), int(height * roi_ratio["y2"]))

    results = model(frame, verbose=False)[0]
    current_time = time.time()
    vehicles_in_roi = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id in [2, 3, 5, 7] and conf > 0.3:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            center_key = (round(cx, -1), round(cy, -1))

            in_roi = (roi_top_left[0] <= cx <= roi_bottom_right[0] and
                      roi_top_left[1] <= cy <= roi_bottom_right[1])

            if in_roi:
                if center_key not in vehicle_times:
                    vehicle_times[center_key] = {'start_time': current_time, 'counted': False}
                duration = current_time - vehicle_times[center_key]['start_time']
                vehicles_in_roi.append(center_key)

                # 경고 기준 시간: 1초
                if duration >= 1 and not vehicle_times[center_key]['counted']:
                    danger_count += 1
                    vehicle_times[center_key]['counted'] = True

                # 경고 표시
                if duration >= 1:
                    color = (0, 0, 255)
                    thickness = 5
                    label = "VERY CLOSE!" if area >= 10000 else "WARNING!"
                else:
                    color = (0, 165, 255)
                    thickness = 3
                    label = f"{duration:.1f}s in ROI"

            else:
                if center_key in vehicle_times:
                    del vehicle_times[center_key]
                label = f"Vehicle {conf:.2f}"
                color = (255, 255, 0)
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # ROI 안에 더 이상 없는 차량 제거
    vehicle_times = {k: v for k, v in vehicle_times.items() if k in vehicles_in_roi}

    # 위험 횟수 출력
    cv2.putText(frame, f"Danger Count: {danger_count}", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)

    # 출력
    frame_resized = cv2.resize(frame, (1280, 720))
    cv2.imshow("YOLO Blind Spot Detection", frame_resized)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
