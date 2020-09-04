import os # 로컬에서 파일을 가져다 쓸 수 있도록 도와줌
import time
import matplotlib.image as mpimg # 시각화 모듈
import matplotlib.pyplot as plt # 시각화 모듈
import tensorflow as tf # 모델 학습을 위한 모듈 텐서플로우
from tensorflow.keras.optimizers import RMSprop # 모델 컴파일을 위한 모듈
from tensorflow.keras.preprocessing.image import ImageDataGenerator # 이미지 정제하기 위한 모듈
from keras.callbacks import EarlyStopping # 모델 학습 과적합 방지하기 위한 모듈

def start():
    print('이미지 경로 설정', 1)
    # 기본 경로
    base_dir = '/home/dasol/snap/machine_learning/' # 이미지 데이터를 담아둔 경로

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # 훈련에 사용되는 음식/사람 이미지 경로
    train_food_dir = os.path.join(train_dir, 'food')
    train_person_dir = os.path.join(train_dir, 'person')
    print(train_food_dir)
    print(train_person_dir)

    # 테스트에 사용되는 음식/사람 이미지 경로
    validation_food_dir = os.path.join(validation_dir, 'food')
    validation_person_dir = os.path.join(validation_dir, 'person')
    print(validation_food_dir)
    print(validation_person_dir)

    print('이미지 경로 설정 확인', 2)
    time.sleep(3)

    train_food_fnames = os.listdir(train_food_dir)
    train_person_fnames = os.listdir(train_person_dir)

    print(train_food_fnames[:5])
    print(train_person_fnames[:5])
    print('Total training food images :', len(os.listdir(train_food_dir)))
    print('Total training person images :', len(os.listdir(train_person_dir)))

    print('Total validation food images :', len(os.listdir(validation_food_dir)))
    print('Total validation person images :', len(os.listdir(validation_person_dir)))

    print('이미지 확인', 3)
    time.sleep(3)

    nrows, ncols = 4, 4
    pic_index = 0

    fig = plt.gcf()
    fig.set_size_inches(ncols * 3, nrows * 3)

    pic_index += 8

    next_food_pix = [os.path.join(train_food_dir, fname)
                    for fname in train_food_fnames[pic_index - 8:pic_index]]

    next_person_pix = [os.path.join(train_person_dir, fname)
                    for fname in train_person_fnames[pic_index - 8:pic_index]]

    for i, img_path in enumerate(next_food_pix + next_person_pix):
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')

        img = mpimg.imread(img_path)
        plt.imshow(img)
    # 이미지 보여주기
    # plt.show()

    print('모델 구성하기', 4)
    time.sleep(3)
    # 텐서플로우를 이용한 합성곱 신경망(CNN) 모델 구성하기
    # conv2D : 합성곱층
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # 128
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'), # 1024
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()

    print('모델 컴파일하기', 5)
    time.sleep(3)

    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print('이미지 전처리하기', 6)
    time.sleep(3)

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            batch_size=20,
                                                            class_mode='binary',
                                                            target_size=(150, 150))

    print('모델 훈련하기', 7)
    time.sleep(3)


    # monitor : 관찰하고자 하는 값
    # patience : 만약 10이라고 지정하면 개선이 없는 에포크가 10번째 지속될 경우 학습을 종료
    # min_delta=0 : 손실을 개선으로 정량화할지 여부에 대한 임계 값 (무슨 뜻인지 잘 모르겠으니 0)
    # verbose=0 : 인쇄할 내용 결정 0 기본
    # mode='auto' : 값이 점점 낮아지면 min, 높아지면 max
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0, mode='auto')
    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        validation_steps=100,
                        steps_per_epoch=100,
                        epochs=100,
                        verbose=2,
                        # batch_size=32, # default 배치 크기
                        callbacks=[early_stopping])

    print('모델 저장하기', 8)
    time.sleep(3)

    # 모델 저장
    model.save('/home/dasol/snap/food_person_ver5.h5')

    print('모델 정확도와 손실 확인하기', 9)
    time.sleep(3)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()




if __name__ == '__main__':

    start()