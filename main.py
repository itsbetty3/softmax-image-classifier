from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import json

#參數定義
epoch = 0 #最終世代數
max_epoch = 300 #最大世代數
lr = 0.0001 #learning rate 學習率，命名為lr
input_size = 784 #input測資的size = 28x28
category_num = 3 #類別的數量：1、2、6三種
mini_batch_size = 32 #Mini-Batch中的batch大小
min_mean = 0.8 #錯誤度量的夠小平均值

#將類別標籤轉換成One-Hot Vector
def one_hot_encode(y, category_num):
    #把類別1、2、6分別放入y[0]、y[1]、y[2]
    for i in range(len(y)):
        if y[i] == 1:
            y[i] = 0
        elif y[i] == 2:
            y[i] = 1
        elif y[i] == 6:
            y[i] = 2
    return np.eye(category_num)[y]

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True) #把每行的最大值減去，找數值溢出，axis=1代表水平方向，也就是“行”
    #做softmax公式
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#計算Cross-Entropy
def loss_function(y, a):
    epsilon = 1e-7
    return -np.sum(y * np.log(a+epsilon), axis=1) #a+epsilon是為了防止有log(0)的情況發生，若只取sum，會返回一個向量，取mean可以算出平均損失

#打開兩個測資檔
with open("./train.json" , mode = 'r' ) as file:
    train = json.load(file)
    
with open("./test.json" , mode = 'r' ) as file:
    test = json.load(file)
    
np.random.seed(0) #設定亂數種子
random.shuffle(train)
   
xtrain = [] #xtrain[]放入train中Image中的資料
ytrain = [] #ytrain[]放入train中Label中的資料
xtest = [] #xtest[]放入test中Image的資料

#讀取檔案中的資料
for i in range(len(train)):
    ytrain.append(int(train[i]['Label']))
    xtrain.append(train[i]['Image'])
    
for i in range(len(test)):
    xtest.append(test[i]['Image'])

#將train測資中，80%給xtrain和ytrain，用來訓練模型；20%xvali和yvali，用來驗證模型
xtrain, xvali, ytrain, yvali = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)
xtrain = np.array(xtrain).reshape(int(len(train)*0.8), input_size, 1) #把xtrain轉成ndarray
ytrain = one_hot_encode(ytrain, category_num) #ytrain轉成one-hot vector
ytrain = np.array(ytrain).reshape(int(len(train)*0.8), category_num, 1)
yvali = one_hot_encode(yvali, category_num) #yvali轉成one-hot vector
yvali = np.array(yvali).reshape(int(len(train)*0.2), category_num, 1)
xvali = np.array(xvali).reshape(int(len(train)*0.2), input_size, 1) #把xvali轉成ndarray
xtest = np.array(xtest).reshape(len(test), input_size, 1) #把xtest轉成ndarray

#初始化w和b
w = np.random.randn(category_num, input_size) * 0.01
b = np.zeros((category_num, 1))

#迴圈中使用到的參數
batch_num = xtrain.shape[0] // mini_batch_size #每個世代中會跑的batch數量
vbatch_num = xvali.shape[0] // mini_batch_size
train_loss = [] #儲存train資料得到的cross-entropy
vali_loss = [] #儲存validation資料得到的cross-entropy
stop_t_a = [] #把每一次世代跑完，train的資料的準確率放進去
stop_v_a = [] #把每一次世代跑完，validation的資料的準確率放進去
test_result = [] #存放為test測資預測的類別
noimprove_ct = 0
train_true = 0
vali_true = 0
best_vloss = float('inf')

for e in range(max_epoch):
    e_loss = 0 #暫時存放世代中的cross-entropy
    for i in range(batch_num):
        #取出mini_batch_size大小的資料量做計算
        xbatch = xtrain[i * mini_batch_size:(i+1) * mini_batch_size] 
        ybatch = ytrain[i * mini_batch_size:(i+1) * mini_batch_size] 
        
        for j in range(mini_batch_size):
            #做softmax function
            neurons = np.matmul(w, xbatch[j]) + b 
            a = softmax(neurons)

            delta = ybatch[j] - a #計算delta值
            #xbatch_T = np.transpose(xbatch, (0, 2, 1))
            grad_w = np.matmul(delta, xbatch[j]) / mini_batch_size #計算w梯度值
            grad_b = np.sum(delta, axis=0, keepdims=True) / mini_batch_size #計算b梯度值
            
            w -= lr * grad_w
            b -= lr * grad_b
            
    for i in range(batch_num):
        #取出mini_batch_size大小的資料量做計算
        xbatch = xtrain[i * mini_batch_size:(i+1) * mini_batch_size] 
        ybatch = ytrain[i * mini_batch_size:(i+1) * mini_batch_size] 
        
        #做softmax function
        neurons = np.matmul(w, xbatch) + b 
        a = softmax(neurons)
        
        #計算cross-entropy的值
        loss = np.mean(loss_function(ybatch, a))
        e_loss += loss #負責存放此mini_batch_size中cross-entropy的累加值
        
    #train_loss
    e_loss /= batch_num #取出此世代的cross-entropy平均值
    train_loss.append(e_loss) #存入名為train_loss的list中
        
    for i in range(vbatch_num):
        x_vbatch = xvali[i * mini_batch_size:(i+1) * mini_batch_size] 
        y_vbatch = yvali[i * mini_batch_size:(i+1) * mini_batch_size]
        
        vali_neurons = np.matmul(w, x_vbatch) + b #使用此世代的w和b計算softmax function
        vali_a = softmax(vali_neurons)
        temp_vloss = np.mean(loss_function(y_vbatch, vali_a))  
        
    #vali_loss
    temp_vloss /= batch_num
    vali_loss.append(temp_vloss) #存入名為vali_loss的list中
    
    #設立終止條件
    if e > 0 and train_loss[-1] < train_loss[-2] and vali_loss[-1] > vali_loss[-2]: #當世代數增加，若訓練資料集準確率上升，而未參與訓練的驗證集準確率卻下降，則可停止訓練
        print(1)
        break
    
    if e_loss < min_mean: #訓練集的錯誤度量(cross-entropy)達到足夠小的平均值
        print(2)
        break
        
    epoch += 1
    
#train_accuracy
train_acu_neurons = np.matmul(w, xtrain) + b #使用此世代的w和b計算softmax function，得到訓練準確率
train_acu_a = softmax(train_acu_neurons)
train_predict = np.argmax(train_acu_a, axis=1) #找到每行最大的值的索引位置
train_accuracy = np.sum(train_predict == np.argmax(ytrain, axis=1)) / len(ytrain) * 100 #比對train_acu_a中最大值的位置是否等於ytrain中最大值的位置，並計算百分率，得到訓練準確率
print(train_acu_a)
print(train_predict)
print(train_acu_a)
#vali_accuracy
vali_acu_neurons = np.matmul(w, xvali) + b #使用此世代的w和b計算softmax function
vali_acu_a = softmax(vali_acu_neurons)
vali_predict = np.argmax(vali_acu_a, axis=1) #找到每行最大的值的索引位置
vali_accuracy = np.sum(vali_predict == np.argmax(yvali, axis=1)) / len(yvali) * 100 #比對vali_a中最大值的位置是否等於yvali中最大值的位置，並計算百分率，得到驗證準確率
print(vali_accuracy)
print(yvali)
print(vali_acu_a)

#處理圖片
plt.plot(range(1, len(train_loss)+1), train_loss) #第一條線為train的世代Cross Entropy
plt.plot(range(1, len(vali_loss)+1), vali_loss) #第二條線為validation的世代Cross Entropy
plt.xlabel('Epoch') #設定x軸的label
plt.ylabel('Error') #設定y軸的label
plt.legend(["train cross entropy", "validation cross entropy"], loc = "lower right")
#plt.savefig('./ouput.png') #儲存圖片
#plt.clf() #關閉圖片
plt.show()
'''
#預估test測資的類別
test_neurons = np.matmul(xtest, w) + b #使用最終的w和b值去計算softmax function
test_a = softmax(test_neurons)
predict_test = np.argmax(test_a, axis=1) #找到每行最大的值的索引位置
#判斷是1、2、6哪個類別
for i in predict_test:
    if i == 0:
        test_result.append(1)
    elif i == 1:
        test_result.append(2)
    elif i == 2:
        test_result.append(6)
        
#將結果存入名為test_output.txt的檔
with open("./test_output.txt", 'w') as file:
    for i in test_result:
        file.write(f"{i}\n")
        
#輸出資料
print("End Epoch: ", end = '')
print(epoch)
print("Learning rate: ", end = '')
print(lr)
print("Train Accuracy: ", end = '')
print(f"{train_accuracy:.2f}", end = '%')
print("\nValidation Accuracy: ", end = '')
print(f"{vali_accuracy:.2f}", end = '%\n')
'''
print(w)
print(b)