import cv2
import numpy as np
import serial
import RPi.GPIO as GPIO

arduino = serial.Serial('/dev/ttyACM0', 9600)
cap = cv2.VideoCapture(0)

GPIO.setmode(GPIO.BOARD)
GREEN_PIN = 11
RED_PIN = 12
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(RED_PIN, GPIO.OUT)

def detectColor(frame: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return cv2.bitwise_and(frame, frame, mask=mask)

def detectContourSize(image: np.ndarray, threshold: int) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        if max(cv2.contourArea(cnt) for cnt in contours) > threshold:
            return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    green = detectColor(frame, np.array([40, 100, 100]), np.array([75, 255, 255]))
    red = detectColor(frame, np.array([0, 100, 100]), np.array([20, 255, 255]))

    green_detected = detectContourSize(green, 300)
    red_detected = detectContourSize(red, 50)

    GPIO.output(GREEN_PIN, green_detected)
    GPIO.output(RED_PIN, red_detected)

    print(f"Green detected: {green_detected}, Red detected: {red_detected}")

    cv2.imshow("Green", green)
    cv2.imshow("Red", red)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
