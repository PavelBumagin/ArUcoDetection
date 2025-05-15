import cv2
import numpy as np
import glob

# Размер шахматной доски
chessboard_size = (9, 6)
square_size = 0.025  # размер квадрата в метрах

# Генерация координат точек в 3D (Z=0, потому что доска плоская)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

# Массивы для хранения точек
objpoints = []  # 3D точки
imgpoints = []  # 2D точки

# Загрузка всех изображений
images = glob.glob('calibration_images/*.jpg')  # Путь к папке с фото

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Поиск углов
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Отрисовка углов
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Калибровка
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# Сохранение параметров
np.savez('camera_calibration.npz',
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs)

print("Калибровка завершена.")
print("Матрица камеры:\n", camera_matrix)
print("Коэффициенты искажения:\n", dist_coeffs)