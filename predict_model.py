import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

# load data
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# x_train
X_train_feature = []
for i in range(len(X_train)):
    feature = hog(X_train[i],orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),block_norm="L2")
    X_train_feature.append(feature)
X_train_feature = np.array(X_train_feature,dtype = np.float32)

# cho x_test
X_test_feature = []
for i in range(len(X_test)):
    feature = hog(X_test[i],orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),block_norm="L2")
    X_test_feature.append(feature)
X_test_feature = np.array(X_test_feature,dtype=np.float32)

model = LinearSVC(C=10)
model.fit(X_train_feature,y_train)
y_pre = model.predict(X_test_feature)
print('Accuracy =',accuracy_score(y_test,y_pre))

# Đọc ảnh
n = 4 # number of images

for i in range(20, 20+n):
    # Read image
    image = cv2.imread(f"segmented_image/img2/imag{i}.png")

# Chuyển đổi sang ảnh xám
im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Làm mờ ảnh
im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Áp dụng ngưỡng nhị phân
_, thre = cv2.threshold(im_blur, 0, 255, cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)

# Thêm đệm cho ảnh
thre = np.pad(thre, (20, 20), 'constant', constant_values=(0, 0))

# Thay đổi kích thước ảnh về 28x28
thre_resized = cv2.resize(thre, (28, 28), interpolation=cv2.INTER_AREA)

# Làm dày ảnh
thre_dilated = cv2.dilate(thre_resized, (3, 3))

# Trích xuất đặc trưng HOG
feature = hog(thre_dilated, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2")

# Dự đoán nhãn cho ảnh
prediction = model.predict(feature.reshape(1, -1))

# In ra kết quả dự đoán
print("Kết quả dự đoán:", prediction[0])

cv2.waitKey(0)