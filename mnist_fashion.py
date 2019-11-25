#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt

    
class MLP(tf.keras.Model):
    def __init__(self, pixel_size):
        
        self.pixel_size = pixel_size
        self.hidden_size = 128
        self.batch_size = 64
        self.num_class = 10
        
        super(MLP, self).__init__()
        self.W1 = tf.Variable(tf.random.truncated_normal([self.pixel_size, self.hidden_size], 
                                                         stddev = 0.1))
        self.b1 = tf.Variable(tf.random.truncated_normal([self.hidden_size], stddev = 0.1))
        self.W2 = tf.Variable(tf.random.truncated_normal([self.hidden_size, self.num_class], stddev = 0.1))
        self.b2 = tf.Variable(tf.random.truncated_normal([self.num_class], stddev = 0.1))
        
    def call(self, inputs):
        output1 = tf.add(tf.linalg.matmul(inputs, self.W1), self.b1)
        final_output1 = tf.nn.relu(output1)
        logits = tf.add(tf.linalg.matmul(final_output1, self.W2), self.b2)
        
        return logits
    
    def loss(self, logits, labels):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
        return tf.reduce_mean(loss)
    
    def accuracy(self, logits, labels,):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def get_data():
    ((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
    trainX = np.array(np.reshape(trainX, [trainX.shape[0],-1]) / 255, dtype=np.float32)
    testX = np.array(np.reshape(testX, [testX.shape[0],-1]) / 255,dtype=np.float32)
    trainY = tf.one_hot(trainY, depth=10)
    testY = tf.one_hot(testY, depth=10)

    return ((trainX, trainY), (testX, testY))

def train(model, train_inputs, train_labels):
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
    indices = tf.range(start=0, limit=tf.shape(train_inputs)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    train_inputs = tf.gather(train_inputs, shuffled_indices)
    train_labels = tf.gather(train_labels, shuffled_indices)
    
    for x in range(0, train_inputs.shape[0], model.batch_size):
        batch_inputs = train_inputs[x: x + model.batch_size, :]
        batch_labels = train_labels[x: x + model.batch_size]
        
        # Optimize gradients
        with tf.GradientTape() as tape:
            predictions = model.call(batch_inputs)
            loss = model.loss(predictions, batch_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        
def test(model, test_inputs, test_labels):
    acc = []
    for x in range(0, test_inputs.shape[0], model.batch_size):
        batch_inputs = test_inputs[x: x + model.batch_size, :]
        batch_labels = test_labels[x: x + model.batch_size]
        
        predictions = model.call(batch_inputs)
        acc.append(model.accuracy(predictions, batch_labels))
    
    return np.mean(acc)

((trainX, trainY), (testX, testY)) = get_data()
model = MLP(784)

epochs = 5
for i in range(epochs):
    train(model, trainX, trainY)
    test_acc = test(model, testX, testY)
    print(test_acc)

print("Accuracy on test set after {} epochs steps: {}".format(epochs, test_acc))
    


# In[2]:


def calculate_r(model, input):
    W1 = model.W1
    W2 = model.W2
    
    original = tf.reshape(input,[-1, model.pixel_size])
    x = tf.reshape(input,[-1, model.pixel_size])
    k_pred = tf.argmax(model(x), 1).numpy()[0]
    k_true = k_pred
    r_out = tf.constant(0,shape=[model.pixel_size],dtype=tf.float32)
    counter = 0
    while k_pred == k_true and counter <= 10000:
        counter += 1
        gradient = tf.matmul(W1,W2)
        delta_w = np.transpose(np.delete(gradient,k_true, axis=1)) - gradient[:,k_true]
    
    
        fx = model(x)
        delta_f = np.delete(fx, k_true)  - fx[:,k_true]
        w_norm = np.sum(np.abs(delta_w)**2, axis = -1)**(0.5)
        standard = np.abs(delta_f)/w_norm
        index = tf.argmin(standard).numpy()
        r = tf.abs(delta_f[index,]) / tf.square(tf.norm(delta_w[index,])) * delta_w[index,]
        x += r
        k_pred = tf.argmax(model(x), 1).numpy()[0]
        r_out += r
    print("iteration used: ", counter)
    
    return (original + r_out*1.02, r_out * 1.02)    


# In[4]:


def adverserial(model, inputs):
    all_x = []
    all_r_out = []
    for i in range(inputs.shape[0]):
        print("attacking image {}".format(i+1))
        (x, r_out) = calculate_r(model, inputs[i,:])
        all_x.append(x)
        all_r_out.append(r_out)
        
    all_x = np.array(all_x)
    all_r_out = np.array(all_r_out)
        
    return (all_x[:,0,:], all_r_out)



# In[18]:


def id_to_category(label_idx):
    lookup = {0: "T-shirt/top",1: "Trouser",2: "Pullover",
             3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt",
              7:"Sneaker", 8:"Bag", 9:"Ankle boot"}
    convert = list(map(lambda x:lookup[x], label_idx))
    return convert

def visualize_results(image, true_labels, pred_labels, file_name):
    num_images = image.shape[0]
    pred_cats = id_to_category(pred_labels)
    true_cats = id_to_category(true_labels)
    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(image[ind], cmap= plt.get_cmap('gray'))
        pl = pred_cats[ind] + str(pred_labels[ind])
        al = true_cats[ind] + str(true_labels[ind])
        ax.set(title="PL: {}\nAL: {}".format(pl, al))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()


# In[19]:

(all_x_noised, all_r_out) = adverserial(model, trainX[:10])
num_img = all_x_noised.shape[0]
original_img = tf.reshape(trainX[:num_img,],[-1,28,28]).numpy()
noised_img = tf.reshape(all_x_noised,[all_x_noised.shape[0],28,28]).numpy()
true_labels=tf.argmax(trainY[:num_img],axis=1).numpy()
pred_labels_original=tf.argmax(model(tf.reshape(trainX[:num_img,],
                [num_img, 784])),axis=1).numpy()
pred_labels_noised=tf.argmax(model(all_x_noised),axis=1).numpy()
visualize_results(original_img, true_labels, pred_labels_original,"original.png")
visualize_results(noised_img, true_labels, pred_labels_noised,"noised.png")


# In[ ]:




