import cv2

# 웹캠 연결
cap = cv2.VideoCapture(0)

# 웹캠 열리는지 확인
if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

print("✅ 웹캠 연결 성공")

prev_gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    # 프레임 사이즈 측정
    height, width, _ = frame.shape

    # 사각지대(Blind Spot) 영역 설정 - 오른쪽 하단 기준
    # 예: 오른쪽 화면 1/4 부분
    roi_top_left = (int(width * 0.75), int(height * 0.4))       # 왼쪽 상단
    roi_bottom_right = (int(width * 0.98), int(height * 0.95))  # 오른쪽 하단

    # ROI만 잘라내기
    roi_frame = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

    #그레이스케일 + 블러
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    motion_detected = False

    if prev_gray is not None:
        # 프레임 차이
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        #윤곽 검출
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 300:
                continue  # 작은 움직임 무시

            motion_detected = True
            break  # 하나만 감지해도 충분
    # 햔재 ROI 프레임 저장
    prev_gray = gray.copy()

    # 사각형으로 ROI 영역 시각화
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 0, 255), 2)
    cv2.putText(frame, 'Blind Spot Zone', (roi_top_left[0], roi_top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 화면 출력
    cv2.imshow('Webcam with Blind Spot', frame)

    # 'q' 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
cv2.destroyAllWindows()
