import keras
import cv2
from keras.models import model_from_json
json_file = open('model_digit.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_digit.h5")
print("Loaded")

img_rows=28
img_cols=28
im=cv2.imread("test.jpg",0)
th, im = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow("binary",im)
#cv2.waitKey(0)
X=im.reshape(1,img_rows,img_cols,1)
input_shape=(img_rows,img_cols,1)
X = X.astype('float32')
X /= 255
print(X.shape)
num_category=10
predict=loaded_model.predict_classes(X)
print(predict)
