<div align="center">
# Road_Object-Detection
</div>

## <div align="center">Documentation</div>

See the for full documentation on training, testing and deployment. See below for quickstart examples.

<details open>
<summary>Install</summary>

## Hi, if you want to see the video of the result of this project, click the link!
YouTube [Link](https://www.youtube.com/watch?v=xjyxl7CHh_0)


Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
>>> git clone https://github.com/AI-Expert-04/Road_Object_Detection.git  # clone
>>> conda create —name Road_Object-Detection-env python=3.8
>>> conda activate Road_Object-Detection-env
Pycharm Termainal >>> pip install -r requirements.txt # install
```

</details>


5. DBB100K data 다운 / DBB100K data [Link](https://drive.google.com/file/d/157GRrqHjiSu8FJegARt-7iNIsdH2NhRh/view?usp=share_link) / Video data [Link](https://drive.google.com/file/d/1ydJfILsKlDDJ7pnUyLlWelOZeta1xQRM/view?usp=share_link) 

6. yolov3 and Ration Net Model 다운 / models [Link](https://drive.google.com/file/d/1-KI-WpQFkRWdidBipCqoMZboaVuBvvjG/view?usp=share_link)


# report [Link](https://docs.google.com/document/d/16T0VQJriU-VXSOssZI7Cu45VG0dLgNY1N7YgtebJXVk/edit?usp=sharing)


image_object_detection

model : MabileNet 
weight : imagenet
Optimzer : RMSprop
data : train_dataset

### yolov 학습
model : MabileNet
weight : imagenet
optimizer : SGD

<pre><code>    model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False,  input_shape=(224, 224, 3))
    model.trainable = False # 1000개의 가중치를 학습하지 않음.

    model = tf.keras.Sequential([
        model, # imageNet 전이 학습 Input_layer
        # Convolution Neural Network (Convolution 신경망)
        tf.keras.layers.Conv2D(1024, (3, 3), padding = 'SAME'), # padding 사용해 필터를 줄임
        tf.keras.layers.Conv2D(1024, (3, 3), padding='SAME'), # padding 한번 더해 필터를 더 줄임
        tf.keras.layers.GlobalAveragePooling2D(), # 필터에 사용될 Parameter 수를 줄여 차원을 감소 즉 2차원
        ## hidden_layer1 ~ hidden_layer2
        # 완전 연결 신경망
        tf.keras.layers.Dense(4096), # 1024 -> 4096(hidden_layer1)
        tf.keras.layers.Dense(735), # 4096(hidden_layer1) -> 735(hidden_layer2)
        tf.keras.layers.Reshape((7, 7, 15)) # 필터를 다시 되돌림. Output_layer
    ]); model.summary()

    if not os.path.exists('../logs'):
        os.mkdir('../logs')

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../logs')

    # SGD(Stochastic Gradient Decent) 확률적 경사 하강법
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.000001, momentum=0.9) # 최적화 함수
    model.compile(loss=yolo_multitask_loss, optimizer=optimizer, run_eagerly=True) # 실패 함수
    model.fit(images, labels, epochs=5, verbose=1, callbacks=[tensorboard]) # 학습
    if not os.path.exists('../models'):
        os.mkdir('../models')

    model.save('../models/yolo_trained.h5')</code></pre>      
    
    
### Retina 학습.
<pre><code>    model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False,  input_shape=(224, 224, 3))
    model.trainable = False # 1000개의 가중치를 학습하지 않음.

    model = tf.keras.Sequential([
        model, # imageNet 전이 학습 Input_layer
        # Convolution Neural Network (Convolution 신경망)
        tf.keras.layers.Conv2D(1024, (3, 3), padding = 'SAME'), # padding 사용해 필터를 줄임
        tf.keras.layers.Conv2D(1024, (3, 3), padding='SAME'), # padding 한번 더해 필터를 더 줄임
        tf.keras.layers.GlobalAveragePooling2D(), # 필터에 사용될 Parameter 수를 줄여 차원을 감소 즉 2차원
        ## hidden_layer1 ~ hidden_layer2
        # 완전 연결 신경망
        tf.keras.layers.Dense(4096), # 1024 -> 4096(hidden_layer1)
        tf.keras.layers.Dense(735), # 4096(hidden_layer1) -> 735(hidden_layer2)
        tf.keras.layers.Reshape((7, 7, 15)) # 필터를 다시 되돌림. Output_layer
    ]); model.summary()

    if not os.path.exists('../logs'):
        os.mkdir('../logs')

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../logs')

    # SGD(Stochastic Gradient Decent) 확률적 경사 하강법
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.000001, momentum=0.9) # 최적화 함수
    model.compile(loss=yolo_multitask_loss, optimizer=optimizer, run_eagerly=True) # 실패 함수
    model.fit(images, labels, epochs=5, verbose=1, callbacks=[tensorboard]) # 학습
    if not os.path.exists('../models'):
        os.mkdir('../models')

    model.save('../models/yolo_trained.h5')</code></pre>      
