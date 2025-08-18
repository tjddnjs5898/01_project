import cv2
import time

# 웹캠 연결
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

print("✅ 웹캠 연결 성공")

prev_gray = None
motion_detected = False
motion_start_time = None
motion_duration = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    height, width, _ = frame.shape

    # 사각지대 영역 설정
    roi_top_left = (int(width * 0.75), int(height * 0.4))      
    roi_bottom_right = (int(width * 0.98), int(height * 0.95))  

    roi_frame = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    current_motion = False  # ✅ 프레임마다 감지 상태 초기화

    if prev_gray is not None:
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 300:
                continue
            current_motion = True
            break

    prev_gray = gray.copy()  # ✅ 다음 프레임 비교를 위한 저장

    # ✅ 시간 측정 로직
    if current_motion:
        if not motion_detected:
            motion_start_time = time.time()
        motion_detected = True
        motion_duration = time.time() - motion_start_time
    else:
        motion_detected = False
        motion_start_time = None
        motion_duration = 0

    # ROI 표시
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 0, 255), 2)
    cv2.putText(frame, 'Blind Spot Zone', (roi_top_left[0], roi_top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 경고 메시지 및 시간 표시
    if motion_detected:
        cv2.putText(frame, 'Beware of blind spots!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        duration_text = f"Duration: {motion_duration:.1f} sec"
        cv2.putText(frame, duration_text, (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        # 3초 이상 머물면 추가 경고 문구 출력
        if motion_duration >= 1.5:
            cv2.putText(frame, 'Be careful!', (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


    cv2.imshow('Side mirror camera system', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
