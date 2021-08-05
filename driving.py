import scipy.misc
import random
import cv2

X = []
Y = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0


#read data.txt

with open('driving_dataset/data.txt') as f:
    for line in f:
        X.append('driving_dataset/' + line.split()[0])
        #the steering wheel angle in radians is used as the output

        Y.append(float(line.split()[1]) * scipy.pi/180)

#get number of images

num_images = len(X)

train_x = X[:int(len(X) * 0.8)]
train_y = Y[:int(len(Y) * 0.8)]

val_x   = X[-int(len(X) * 0.2):]
val_y   = Y[-int(len(Y) * 0.2):]


num_train_images = len(train_x)
num_val_images = len(val_x)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(train_x[(train_batch_pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)
        y_out.append([train_y[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out


def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(val_x[(val_batch_pointer + i) % num_val_images])[-150:], [66, 200]) / 255.0)
        y_out.append([val_y[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out