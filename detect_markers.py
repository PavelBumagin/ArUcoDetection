import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R


def rvec_to_euler_angles(rvec):
    rot = R.from_rotvec(rvec)
    euler = rot.as_euler('zyx', degrees=True) # Yaw-Pitch-Roll (по Z, Y, X)
    return euler  # три угла в градусах


# ==== Параметры камеры  ====
camera_matrix = np.array([[943.14025966, 0, 647.95301629],
                          [0, 907.76452814, 322.79856978],
                          [0,   0,   1]], dtype=np.float64)
dist_coeffs = np.array([-0.1968091,   0.54916959,  0.01793236,  0.00326459, -0.48967739])
# ==== Выбор одного из существующих наборов маркеров ====
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)


# Размер маркера в метрах
marker_length = 0.05

# ==== Камера ====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            # Рисуем рамку и оси координат
            aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03, thickness=2)
            angles = rvec_to_euler_angles(rvecs[i][0])
            # Выводим в консоль позицию и ориентацию
            print(f"ID: {ids[i][0]} | Позиция: {tvecs[i][0]} | Углы поворота: {angles}")

    #выход по нажатию клавиши q
    cv2.imshow('Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#Прекращение захвата изображения, закрытие окна
cap.release()
cv2.destroyAllWindows()