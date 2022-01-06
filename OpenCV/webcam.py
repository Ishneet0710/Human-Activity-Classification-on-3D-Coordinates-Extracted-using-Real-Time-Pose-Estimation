import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()