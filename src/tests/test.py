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

# 상태 변수 초기화
motion_detected = False
motion_start_time = None
motion_duration = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("영상 종료")
        break

    height, width, _ = frame.shape

    # 사각지대 ROI (우측 하단 기준)
    roi_top_left = (int(width * 0.75), int(height * 0.4))
    roi_bottom_right = (int(width * 0.98), int(height * 0.95))

    results = model(frame, verbose=False)[0]
    current_motion = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id in [2, 3, 5, 7] and conf > 0.3:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 차량이 ROI 안에 있는지 확인
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if roi_top_left[0] <= cx <= roi_bottom_right[0] and roi_top_left[1] <= cy <= roi_bottom_right[1]:
                current_motion = True
                # 감지된 차량 박스
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, f"Vehicle ({conf:.2f})", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)

    # 시간 측정 로직
    if current_motion:
        if not motion_detected:
            motion_start_time = time.time()
        motion_detected = True
        motion_duration = time.time() - motion_start_time
    else:
        motion_detected = False
        motion_start_time = None
        motion_duration = 0

    # 경고 메시지
    if motion_detected:
        cv2.putText(frame, f"Time in blind spot: {motion_duration:.1f} sec", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 0), 4)

        if motion_duration >= 3:
            cv2.putText(frame, "Be careful!", (50, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
        else:
            cv2.putText(frame, "Be careful!", (50, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 165, 255), 4)
    else:
        cv2.putText(frame, "safe", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 0), 5)

    # 프레임 크기 조정 및 출력
    frame_resized = cv2.resize(frame, (1280, 720))
    cv2.imshow("YOLO Blind Spot Detection", frame_resized)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
