import cv2
import numpy as np
import mss
import easyocr  # OCR 라이브러리 추가
import re       # 정규표현식(문자열 필터링) 라이브러리 추가
import time     # 시간 측정을 위해 time 모듈 추가
import collections # 최빈값 계산

# ==========================================
# [OCR Load part]
# 숫자만 read -> 'en'(영어) 모델만 로드
reader = easyocr.Reader(['en']) 
print("OCR 로드 완료")

# # 중복 방지를 위한 상태 기억 변수
# last_damage = ""
# last_time = 0

# 상태 기억 변수를 버퍼(리스트) 형태로 저장
damage_buffer = []
last_detect_time = time.time()
# ==========================================


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


        # ==========================================
        # [text OCR Part]
            # OCR로 크롭된 이미지 읽기 (detail=0은 텍스트 문자열만 리스트로 반환함)
            ocr_result = reader.readtext(final_crop, detail=0)

            if ocr_result:
                # 인식된 결과 리스트를 하나의 문자열로 합치기
                raw_text = "".join(ocr_result)

                # 정규표현식을 사용하여 숫자(0-9)가 아닌 모든 문자(알파벳, 쉼표 등) 제거
                # "306,867"을 "306867"의 순수 숫자로 정제합니다.
                clean_number = re.sub(r'[^0-9]', '', raw_text)

                # # 중복 방지
                # if clean_number:
                #     current_time = time.time()
                    
                #     # 방금 읽은 데미지와 다르거나, 같은 데미지라도 1.5초 이상 지났다면 (연속 타격 인정) 새로운 타격으로 간주
                #     if clean_number != last_damage or (current_time - last_time) > 1.5:
                #         print(f"[새로운 데미지 인식 완료] : {clean_number}")
                        
                #         # 방금 인식한 데이터로 기억 갱신
                #         last_damage = clean_number
                #         last_time = current_time


                # 버퍼 수집
                # 자잘한 1~3자리 노이즈는 무시하고, 4자리 이상의 유의미한 데미지만 바구니에 담기
                if clean_number and len(clean_number) >= 4:
                    damage_buffer.append(clean_number)
                    last_detect_time = time.time() # 숫자가 마지막으로 목격된 시간 갱신
        
        # 화면에서 숫자가 안 보인지 0.5초가 지났고, 바구니(버퍼)에 수집된 데이터가 있다면?
        if (time.time() - last_detect_time) > 0.1 and len(damage_buffer) > 0:
            
            # 수집된 숫자들 중 '가장 많이 등장한 숫자'를 찾음 (노이즈 필터링의 핵심)
            counter = collections.Counter(damage_buffer)
            final_damage = counter.most_common(1)[0][0]
            
            # 추후 이 final_damage를 SQLite DB에 INSERT 하시면 됩니다!
            print(f" [최종 데미지 확정 DB 저장] : {final_damage} (참고: 수집된 프레임 수 {len(damage_buffer)}개)")
            print("-" * 50)
            
            # 다음 타격을 위해 바구니를 깨끗하게 비우기
            damage_buffer.clear()
            
        # ==========================================

        # 화면 출력 테스트(불필요한 화면 주석처리)
        # cv2.imshow('Original ROI', img_bgr)       
        # cv2.imshow('Yellow Mask', mask_cleaned)
        cv2.imshow('Bounding Box Result', img_result)

        # 'q' 키를 누르면 루프 탈출
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# 루프 종료 후 열려있는 OpenCV 창 모두 닫기
cv2.destroyAllWindows()