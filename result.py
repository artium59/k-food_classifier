import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def draw_graph(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def image_test(model_path, image_path, label_path='label.npy'):
    import numpy as np

    from tensorflow.keras import models
    from PIL import Image

    label = np.load(label_path, allow_pickle=True)

    test_image = Image.open(image_path)  # type: Image.Image
    resize_test_image = np.array(test_image.resize((299, 299))).reshape((1, 299, 299, 3)) / 255.

    # print(resize_test_image.shape)
    model = models.load_model(model_path)
    prediction = model.predict(resize_test_image)

    plt.imshow(test_image)
    plt.title(label[np.argmax(prediction[0])],
              fontproperties=fm.FontProperties(fname='font/S-Core_Dream_OTF/SCDream5.otf', size=18))
    plt.show()


def video_test():
    pass


if __name__ == '__main__':
    image_test(model_path='train_model/k-food_valacc8530.h5',
               # image_path='kfood/밥/김치볶음밥/Img_070_0001.jpg',
               image_path='12345.jpg')
