import cv2

# 웹캠 연결
cap = cv2.VideoCapture(0)

# 웹캠 열리는지 확인
if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

print("✅ 웹캠 연결 성공")

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
