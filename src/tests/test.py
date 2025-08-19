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

motion_detected = False
motion_start_time = None
motion_duration = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("영상 종료")
        break

    results = model(frame, verbose=False)[0]
    current_motion = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # 차량 클래스만 감지 (car, motorbike, bus, truck)
        if cls_id in [2, 3, 5, 7] and conf > 0.3:
            current_motion = True

            # 감지된 차량에 사각형 그리기
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Vehicle ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 감지 시간 로직
    if current_motion:
        if not motion_detected:
            motion_start_time = time.time()
        motion_detected = True
        motion_duration = time.time() - motion_start_time
    else:
        motion_detected = False
        motion_start_time = None
        motion_duration = 0

    # 경고 메시지 출력
    if motion_detected:
        cv2.putText(frame, 'Beware of vehicles!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(frame, f'Duration: {motion_duration:.1f} sec', (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if motion_duration >= 1.5:
            cv2.putText(frame, 'Be careful!', (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    else:
        cv2.putText(frame, 'Safe', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    frame_resized = cv2.resize(frame, (960, 540))
    cv2.imshow("YOLO Vehicle Detection (Dynamic Boxes)", frame_resized)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
