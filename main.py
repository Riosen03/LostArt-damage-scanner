import cv2
import numpy as np
import mss

# 화면 캡처를 위한 mss 객체 생성
with mss.mss() as sct:
    # ==============================================
    # [ROI part]

    # ROI 지정
    monitor = {"top": 250, "left": 600, "width": 800, "height": 600}    # 해당 위치는 직접 조절 필요
    print("종료 - 'q'")

    # 실시간 캡처 무한 루프 
    while True:
        # 지정된 ROI 캡처 
        sct_img = sct.grab(monitor)

        # 캡처된 화면을 NumPy 배열로 변환 / BGRA 포맷을 OpenCV 처리를 위해 BGR 포맷으로 변환 (알파 채널 제거)
        img_bgr = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

    # ==========================================


        # ==========================================
        # [Preprocessing part]

        # BGR을 HSV 색상 공간으로 변환
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 크리티컬 폰트(노란색)의 HSV 범위 지정
        lower_yellow = np.array([15, 150, 150])
        upper_yellow = np.array([35, 255, 255])

        # 노란색 영역만 흰색(255), 나머지는 검은색(0)으로 만드는 마스크 생성
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Morphology
        # 3x3 kernel 생성
        kernel = np.ones((3, 3), np.uint8)
        
        # Closing 연산 -> antialiasing 테두리 복원 및 틈새 메우기
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 윤곽선 검출 & bounding 박스
        # 마스크 이미지에서 흰색 덩어리들의 외곽선(Contours) 찾기
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 원본 이미지에 박스를 그리기 위해 복사본 생성
        img_result = img_bgr.copy()

        # 유효한 윤곽선(숫자 조각)들의 좌표를 담을 리스트
        valid_rects = []

        for cnt in contours:
            # 면적이 50 픽셀 이상인 덩어리(노이즈 제외)만 수집
            if cv2.contourArea(cnt) > 50:
                valid_rects.append(cv2.boundingRect(cnt))

        # 유효한 숫자 조각이 하나라도 발견되었다면 (데미지가 떴다면)
        if valid_rects:
            # 모든 조각을 포함하는 가장 바깥쪽 좌표 계산
            min_x = min([x for x, y, w, h in valid_rects])
            min_y = min([y for x, y, w, h in valid_rects])
            max_x = max([x + w for x, y, w, h in valid_rects])
            max_y = max([y + h for x, y, w, h in valid_rects])

            # 원본 확인용 창에 거대한 하나로 합쳐진 초록색 박스 그리기
            cv2.rectangle(img_result, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

            # OCR에 넘겨주기 위해 mask(흑백) Crop
            # 글자가 테두리에 너무 딱 붙지 않도록 Padding
            pad = 5
            crop_y1 = max(0, min_y - pad)
            crop_y2 = min(mask_cleaned.shape[0], max_y + pad)
            crop_x1 = max(0, min_x - pad)
            crop_x2 = min(mask_cleaned.shape[1], max_x + pad)

            # 최종적으로 잘라낸 흑백 숫자 이미지
            final_crop = mask_cleaned[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # 잘라낸 최종 이미지를 화면에 띄워 확인
            cv2.imshow('Final Crop for OCR', final_crop)

        # ==========================================

        # 화면 출력 테스트
        cv2.imshow('Original ROI', img_bgr)
        cv2.imshow('Yellow Mask', mask_cleaned)
        cv2.imshow('Bounding Box Result', img_result)

        # 'q' 키를 누르면 루프 탈출
        # waitKey(1)은 1ms 동안 키 입력을 대기하며 화면을 갱신하는 역할을 합니다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# 루프 종료 후 열려있는 OpenCV 창 모두 닫기
cv2.destroyAllWindows()