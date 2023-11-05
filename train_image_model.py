import tensorflow as tf
from keras.layers import RandomFlip
from keras.optimizers import Lion

from model import CNeXt

# tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
print(f"Global dtype policy: {tf.keras.mixed_precision.global_policy()}")
# tf.config.run_functions_eagerly(True)

train_data = 'datasets/train'
val_data = 'datasets/valid'
image_size = (256, 256)
batch_size = 256

if __name__ == '__main__':
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_data,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        val_data,
        image_size=image_size,
        batch_size=batch_size,
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"Total classes: {num_classes}")
    with open('models/num_classes.txt', 'w') as f:
        f.write(str(num_classes))

    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal_and_vertical"),
        # RandomRotation(0.2),
    ], name='data_augmentation')

    model = tf.keras.Sequential([
        data_augmentation,
        CNeXt(num_classes=num_classes),
    ])
    model.build(input_shape=(batch_size, *image_size, 3))

    model.compile(optimizer=Lion(learning_rate=0.001), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'], jit_compile=True)
    model.summary()

    model.fit(train_dataset, epochs=5, validation_data=val_dataset)
    model.layers[-1].save_weights('./models/image-model-weights.h5')
