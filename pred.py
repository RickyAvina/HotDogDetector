import PIL
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

img = load_img('data/test/not_hot_dog/plane-img_13.jpg')
img = img.resize((150, 150))
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

model = load_model('models/save_at_30.h5')
print(model.summary())

predictions = model.predict(img_array)
score = predictions[0]

plt.imshow(img)
plt.title("{}% hot dog\n{}% not hot dog".format(100*score[0], 100*(score[1])))
plt.show()

# print("This image is ")
# print("This image is %.2f percent hot dog and %.2f percent not hot dog." % (100 * (1 - score), 100 * score))
