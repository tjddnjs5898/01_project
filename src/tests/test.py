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

# 차량 추적용 변수
# key: (center_x, center_y), value: (entry_time, counted_flag)
vehicle_times = {}
danger_count = 0  # 경고 횟수 누적

# 깜빡이기 위한 타이머
blink_on = True
blink_interval = 0.5  # 초 단위
last_blink_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("영상 종료")
        break

    height, width, _ = frame.shape
    roi_top_left = (int(width * roi_ratio["x1"]), int(height * roi_ratio["y1"]))
    roi_bottom_right = (int(width * roi_ratio["x2"]), int(height * roi_ratio["y2"]))

    # YOLO 객체 감지
    results = model(frame, verbose=False)[0]
    current_time = time.time()
    vehicles_in_roi = []  # 이번 프레임에 감지된 차량 key들

    show_warning = False
    warning_area_large = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # 자동차, 버스, 트럭만
        if cls_id in [2, 3, 5, 7] and conf > 0.3:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 중심 좌표를 반올림해서 차량 추적
            center_key = (round(cx, -1), round(cy, -1))

            # ROI 안에 있는지 확인
            in_roi = (roi_top_left[0] <= cx <= roi_bottom_right[0] and
                      roi_top_left[1] <= cy <= roi_bottom_right[1])

            if in_roi:
                if center_key not in vehicle_times:
                    vehicle_times[center_key] = (current_time, False)

                entry_time, counted = vehicle_times[center_key]
                duration = current_time - entry_time
                vehicles_in_roi.append(center_key)

                if duration >= 1:  # 1초 이상 머무르면 경고
                    show_warning = True
                    if area >= 10000:
                        warning_area_large = True

                    if not counted:
                        danger_count += 1
                        vehicle_times[center_key] = (entry_time, True)

                label = f"{duration:.1f}s in ROI"
                color = (0, 165, 255)
                thickness = 3
            else:
                # ROI 밖 차량은 기록 삭제
                if center_key in vehicle_times:
                    del vehicle_times[center_key]
                label = f"Vehicle {conf:.2f}"
                color = (255, 255, 0)
                thickness = 2

            # 박스 및 텍스트 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # 이번 프레임에 없는 차량은 vehicle_times에서 삭제
    vehicle_times = {k: v for k, v in vehicle_times.items() if k in vehicles_in_roi}

    # 깜빡임 업데이트
    if current_time - last_blink_time >= blink_interval:
        blink_on = not blink_on
        last_blink_time = current_time

   

    # 경고 횟수 출력
    cv2.putText(frame, f"Danger Count: {danger_count}", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)

    # 프레임 출력
    frame_resized = cv2.resize(frame, (1280, 720))
    cv2.imshow("YOLO Blind Spot Detection (Multiple Vehicles)", frame_resized)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
