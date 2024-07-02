import cv2, os, random, time
import numpy as np
from ultralytics import YOLO

# ウェブカメラを起動
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# YOLOv8 モデルをロード
model = YOLO(os.path.join(os.path.dirname(__file__), "weight.pt"))

# 識別されるクラスの名前
classNames = ['Paper', 'Rock', 'Scissors']

# 状態管理
currentScore = 0
timeStamp = time.time()
isActive = False
orderHand = 0
orderWin = 0
gameRecord = np.zeros((2, 3))
def argmax(a):
    i = np.argmax(a)
    return i if a[i] / np.sum(a) > 0.8 else -1

while True:
    success, img = cap.read()
    img = cv2.flip(img, +1)
    results = model(img, stream=True)

    # 検出された手の位置と形を記録
    detectedHands = []

    for r in results:
        for box in r.boxes:
            # 境界を取得
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 検出された手の種類 (0, 1, 2) を取得
            cls = int(box.cls[0])

            # 手の位置と形を記録
            detectedHands.append((x1, cls))

            # 境界を描画し、手の形を表示
            cv2.rectangle(img, (x1, y1), (x2, y2), (104, 3, 217), 5)
            cv2.putText(img, classNames[cls], [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 2, (15, 196, 241), 4)

    # スコア表示
    cv2.putText(img, f'currentScore: {currentScore}', [25, 50], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if isActive:
        # 指示を表示
        hand_s = ['Left', 'Right'][orderHand]
        win_s = ['lose', 'tie', 'win'][orderWin + 1]
        cv2.putText(img, hand_s + ' must ' + win_s, [25, 100], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if len(detectedHands) == 2:
            # 2 つの手を認識したら、それぞれの情報を記録
            detectedHands.sort()
            _, l = detectedHands[0]
            _, r = detectedHands[1]
            gameRecord *= 0.8
            gameRecord[0][l] += 1
            gameRecord[1][r] += 1

            # 正しい手を一定時間出せているかどうかの判定
            if time.time() - timeStamp > 2.0:
                a = argmax(gameRecord[orderHand])
                b = argmax(gameRecord[1 - orderHand])
                if a != -1 and b != -1 and (a + orderWin) % 3 == b:
                    currentScore += 1
                    timeStamp = time.time()
                    isActive = False
                    gameRecord = np.zeros((2, 3))
    else:
        # 次の指示を待っている旨を表示
        cv2.putText(img, 'Ready...' if currentScore == 0 else 'Great!', [25, 100], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 255), 2)

        # 前回のクリアから 0.7 秒経ったら次の指示を準備
        if time.time() - timeStamp > 0.7:
            isActive = True
            while True:
                new_hand = random.randint(0, 1)
                new_win = random.randint(-1, 1)

                # 同じ手のままクリアできないような指示を生成
                if (new_hand == orderHand and new_win != orderWin) or (new_hand != orderHand) and new_win != -orderWin:
                    orderHand = new_hand
                    orderWin = new_win
                    break

    # 画像を表示
    cv2.imshow('Webcam', img)

    # q が入力されたらプログラムを終了する
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()