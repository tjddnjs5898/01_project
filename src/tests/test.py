import cv2

# 0번은 기본 웹캠 (내장형), 외부 카메라를 연결했을 경우 1, 2로 바꿔보세요
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

print("✅ 웹캠 연결 성공. 영상 스트리밍 시작...")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    # 프레임 출력
    cv2.imshow('Webcam Feed', frame)

    # 'q' 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 스트리밍 종료")
        break

# 해제 및 종료
cap.release()
cv2.destroyAllWindows()
