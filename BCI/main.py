import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns
import tensorflow.keras.backend
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

tensorflow.keras.backend.clear_session()
# Feature_all_subject和Label_all_subject路径
path1 = 'Feature_all_subject1.mat'
path2 = 'Label_all_subject.txt'

data = scipy.io.loadmat(path1)
feature = data['Feature_all_subject_norm'][:]
y_label = np.loadtxt(path2)
acc_kFold = []
con_mat_all = np.zeros([4, 4])
feature1_test_acc = []
feature2_recall_value = []
feature3_f1_value = []
feature4_precisionl_value = []

# 采用五折交叉检验进行模型评估
kfold = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kfold.split(feature):
    print("Train:", train_index, "Validation:", test_index)
    x_train, x_test = feature[train_index], feature[test_index]
    y_train, y_test = y_label[train_index], y_label[test_index]

    [m1] = y_train.shape
    [m2] = y_test.shape

    # 定义一些参数
    # 输入大小
    lim = 154
    input_size = (lim, 1)
    feature_input = Input(input_size)
    # 步长
    strides = 1
    # 卷积核大小
    kernel_size1 = 7
    kernel_size2 = 5
    kernel_size3 = 3
    # 卷积核数量
    kernel_num1 = 16
    kernel_num2 = 32
    # 批大小
    batch_size = 100
    # 迭代epoch数量
    epochs_size = 100
    # 添加L2正则化项
    l2_regularization = 0.01
    regularization = l2(l2_regularization)
    x_train = x_train.reshape((m1, lim, 1))
    x_test = x_test.reshape((m2, lim, 1))

    # 卷积神经网络 模型

    # Convolution Layer1
    CNN = Conv1D(kernel_num1,
                 kernel_size1, strides,
                 padding='same',
                 activation='relu',
                 kernel_regularizer=regularization)(feature_input)
    # 标准化
    CNN = BatchNormalization()(CNN)
    # Max Pooling (down-sampling)
    CNN = MaxPooling1D(2)(CNN)
    # Convolution Layer2
    CNN = Conv1D(kernel_num2,
                 kernel_size2,
                 strides,
                 padding='same',
                 activation='relu',
                 kernel_regularizer=regularization)(CNN)
    CNN = BatchNormalization()(CNN)
    # Max Pooling (down-sampling)
    CNN = MaxPooling1D(2)(CNN)
    # Apply Dropout
    CNN = Dropout(0.3)(CNN)
    # Convolution Layer3
    CNN = Conv1D(kernel_num2,
                 kernel_size3,
                 strides, padding='same',
                 activation='relu',
                 kernel_regularizer=regularization)(CNN)
    # Max Pooling (down-sampling)
    CNN = MaxPooling1D(2)(CNN)
    # Apply Dropout
    CNN = Dropout(0.3)(CNN)
    # Fully connected layer
    CNN = Flatten()(CNN)
    CNN = Dense(64, activation='relu')(CNN)
    # Output, class prediction
    output = Dense(4, activation='softmax')(CNN)
    CNN_model = Model(feature_input, output)
    # Adam优化器,自适应学习率的梯度下降算法
    CNN_model.compile(optimizer='Adam',
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
    # 卷积神经网络框架
    CNN_model.summary()

    history = CNN_model.fit(x_train,
                            y_train,
                            epochs=epochs_size,
                            batch_size=batch_size,
                            validation_data=(x_test, y_test))

    # 可视化模型训练过程loss和accuracy
    print(history.history.keys())
    plt.figure(figsize=(4, 4))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.figure(figsize=(4, 4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # 最终测试集准确率
    test_loss, test_acc = CNN_model.evaluate(x_test, y_test)
    acc_kFold.append(test_acc)
    # 计算每一折测试的混淆矩阵
    x_test = x_test.reshape((m2, lim, 1))
    y_pred = CNN_model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=-1)
    con_mat = confusion_matrix(y_test, y_pred)
    # 同时记录总的混淆矩阵
    con_mat_all = con_mat_all + con_mat
    # 计算并打印模型评估指标
    print(test_acc)
    recall_value = recall_score(y_test, y_pred, average='weighted')
    f1_value = f1_score(y_test, y_pred, average='weighted')
    precisionl_value = precision_score(y_test, y_pred, average='weighted')
    print(recall_value)
    print(f1_value)
    print(precisionl_value)
    feature1_test_acc.append(test_acc)
    feature2_recall_value.append(recall_value)
    feature3_f1_value.append(f1_value)
    feature4_precisionl_value.append(precisionl_value)
mean_acc = np.mean(acc_kFold)
con_mat_norm = con_mat_all.astype('float') / con_mat_all.sum(axis=1)[:, np.newaxis]  # 归一化
con_mat_norm = np.around(con_mat_norm, decimals=2)
plt.figure(figsize=(4, 4))
sns.heatmap(con_mat_norm, annot=True, cmap='Blues')
plt.ylim(0, 4)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.show()
print('五折交叉验证---------------------------')
print('识别准确率')
print(mean_acc)
print('test_acc')
print(feature1_test_acc)
print('recall_value')
print(feature2_recall_value)
print('f1_value')
print(feature3_f1_value)
print('precisionl_value')
print(feature4_precisionl_value)
print('-------------------------------------')
