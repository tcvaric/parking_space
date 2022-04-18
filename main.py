import cv2
import pickle
import cvzone
import numpy as np

#Video feed
cap = cv2.VideoCapture('carPark.mp4')

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 65, 33

def checkParkingSpace(imgPro):

    spaceCounter = 0

    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y+height, x:x+width] #画像のトリミング
        count = cv2.countNonZero(imgCrop)

        if count < 400:
            color = (0, 255, 0)
            thickness = 4
            spaceCounter += 1
            cvzone.putTextRect(img, str('Empty'), (x, y + height - 10), scale=1.1,
                               thickness=2, offset=0, colorR=color)
        else:
            color = (0, 0, 255)
            thickness = 3
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        #cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1.3,
                           #thickness=2, offset=0, colorR=color)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(139, 139, 0))

while True:
    #ビデオのループ
    # 現在のフレーム数を取得合計フレームを取得し比較.現在のフレーム数が合計フレーム数に達したら0リセット
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    if not success:
        break

    #グレースケール変換
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ぼかし処理
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    #適応的閾値処理（近傍領域のガウス分布による重み付け平均値）の二値化処理
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 37, 26)
    #平滑化処理,ドットを減らす
    imgMedian = cv2.medianBlur(imgThreshold, 7)
    #カーネルサイズ3*3の1行列の生成
    kernel = np.ones((3, 3), np.uint8)
    #カーネルサイズ3*3の膨張処理
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=2)

    checkParkingSpace(imgDilate)
    cv2.imshow("Image", img)
    #cv2.imshow("ImageBlur", imgBlur)
    #cv2.imshow("ImgThres", imgThreshold)

    cv2.waitKey(20)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()