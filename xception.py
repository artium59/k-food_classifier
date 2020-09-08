import tensorflow as tf

from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception

from result import draw_graph

# GPU Error

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)]
)

##################################################
batch_size = 8

early_stop = EarlyStopping('val_loss', patience=30)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=10, verbose=1)
callbacks = [early_stop, reduce_lr]
##################################################

base_model = Xception(input_shape=(299, 299, 3), include_top=False, weights='imagenet')

base_model.trainable = False

inputs = Input(shape=(299, 299, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(150, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()


# Train

train_datagen = ImageDataGenerator(
                            # featurewise_center=True,
                            # featurewise_std_normalization=True,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            rescale=1./255,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 20
model.fit_generator(train_datagen.flow_from_directory(directory='kfood_299_size/train',
                                                      target_size=(299, 299), batch_size=batch_size,
                                                      class_mode='categorical'),
                    epochs=epochs, verbose=1, callbacks=callbacks,
                    validation_data=test_datagen.flow_from_directory(directory='kfood_299_size/test',
                                                                     target_size=(299, 299), batch_size=batch_size,
                                                                     class_mode='categorical'))

# Fine Tuning

base_model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 20
history = \
    model.fit_generator(train_datagen.flow_from_directory(directory='kfood_299_size/train',
                                                          target_size=(299, 299), batch_size=batch_size,
                                                          class_mode='categorical'),
                        epochs=epochs, verbose=1, callbacks=callbacks,
                        validation_data=test_datagen.flow_from_directory(directory='kfood_299_size/test',
                                                                         target_size=(299, 299), batch_size=batch_size,
                                                                         class_mode='categorical'))

# Save

trained_models_path = 'train_model/'
model_name = trained_models_path + 'k-food' + '.hdf5'
model.save(model_name)

model_name = trained_models_path + 'k-food' + '.h5'
model.save(model_name)

# plot

draw_graph(history)
