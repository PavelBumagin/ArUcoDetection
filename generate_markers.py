import cv2
import matplotlib.pyplot as plt

# Выбираем один из существующих наборов маркеров
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


marker_size = 200  # Размер в пикселях

# Создаем маркеры
for i in range(0,5):
    marker_id = i
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    cv2.imwrite(f'markers/marker_{i}.png', marker_image) #сохраняем изображение в папку markers
    plt.imshow(marker_image, cmap='gray', interpolation='nearest')
    plt.axis('off')  # отключить отображение осей
    plt.title(f'ArUco Marker {marker_id}')
    plt.show()