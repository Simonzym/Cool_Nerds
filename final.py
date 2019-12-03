from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

def preprocess(x_train, y_train, x_test, y_test):
    # normalize the inputs
    x_train = x_train/255.0
    x_train = x_train.astype(np.float32)
    
    x_test = x_test/255.0
    x_test = x_test.astype(np.float32)
    
    # Onehot labels
    y_train = tf.one_hot(y_train, depth = 10)
    
    y_test = tf.one_hot(y_test, depth = 10)
    
    return (x_train, y_train, x_test, y_test)


class MLP(tf.keras.Model):
    def __init__(self, pixel_size):
        
        self.pixel_size = pixel_size
        self.num_classes = 10
        self.hidden_size = 128
        self.batch_size = 64
        
        super(MLP, self).__init__()
        self.W1 = tf.Variable(tf.random.truncated_normal([self.pixel_size, self.hidden_size], stddev = 0.1))
        self.b1 = tf.Variable(tf.random.truncated_normal([self.hidden_size], stddev = 0.1))
        self.W2 = tf.Variable(tf.random.truncated_normal([self.hidden_size, self.num_classes], stddev = 0.1))
        self.b2 = tf.Variable(tf.random.truncated_normal([self.num_classes], stddev = 0.1))
        
    def call(self, inputs):
        output1 = tf.add(tf.linalg.matmul(inputs, self.W1), self.b1)
        relu_output1 = tf.nn.relu(output1)
        logits = tf.add(tf.linalg.matmul(relu_output1, self.W2), self.b2)
        
        return logits
    
    def loss(self, logits, labels):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
        return tf.reduce_mean(loss)
    
    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
  
    
def train(model, train_inputs, train_labels):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    #shuffle
    indices = tf.range(0,train_inputs.shape[0])
    new_indices = tf.random.shuffle(indices)
    new_inputs = tf.gather(train_inputs, new_indices)
    new_labels = tf.gather(train_labels, new_indices)
    
    for x in range(0, new_inputs.shape[0], model.batch_size):
        batch_inputs = new_inputs[x: x + model.batch_size, :]
        batch_labels = new_labels[x: x + model.batch_size]
        
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


(x_train, y_train, x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)
# Reshape the data inputs such that we can put those inputs into MLP
train_inputs = np.reshape(x_train, (-1, 28*28))
test_inputs = np.reshape(x_test, (-1, 28*28))

model = MLP(28*28)
    
epochs = 5
for i in range (epochs):
    train(model, train_inputs, y_train)
    test_acc = test(model, test_inputs, y_test)
    print(test_acc)

print("Accuracy on test set after {} epochs steps: {}".format(epochs, test_acc))


def calculate_r(model, input):
    W1 = model.W1
    W2 = model.W2
    
    original = tf.reshape(input,[-1, model.pixel_size])
    x = tf.reshape(input,[-1, model.pixel_size])
    k_pred = tf.argmax(model(x), 1).numpy()[0]
    k_true = k_pred
    r_out = tf.constant(0,shape=[model.pixel_size],dtype=tf.float32)
    
    counter = 0
    while k_pred == k_true and counter <= 1000:
        counter = counter + 1
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
    
    
    return (original + r_out * 1.02, r_out * 1.02)
    

def adverserial(model, inputs):
    all_x = []
    all_r_out = []
    for i in range(inputs.shape[0]):
        print(i)
        (x, r_out) = calculate_r(model, train_inputs[i,:])
        all_x.append(x)
        all_r_out.append(r_out)
        
    all_x = np.array(all_x)
    all_r_out = np.array(all_r_out)
        
    return (all_x[:,0,:], all_r_out)


(all_x, all_r_out) = adverserial(model, train_inputs[:640,:])
(test_all_x, test_all_r_out) = adverserial(model, test_inputs[:128,:])


#accuracy on the original model
test_logits = model.call(test_all_x)
test_accuracy = model.accuracy(test_logits, y_test[:128])
print("After fool the MLP, the accuracy on the original model becomes", test_accuracy)





class regu_MLP(tf.keras.Model):
    def __init__(self, pixel_size):
        
        self.pixel_size = pixel_size
        self.num_classes = 10
        self.hidden_size = 128
        self.batch_size = 64
        self.c = 25
        self.d = 5
        self.l = 5
        
        super(regu_MLP, self).__init__()
        self.W1 = tf.Variable(tf.random.truncated_normal([self.pixel_size, self.hidden_size], stddev = 0.1))
        self.b1 = tf.Variable(tf.random.truncated_normal([self.hidden_size], stddev = 0.1))
        self.W2 = tf.Variable(tf.random.truncated_normal([self.hidden_size, self.num_classes], stddev = 0.1))
        self.b2 = tf.Variable(tf.random.truncated_normal([self.num_classes], stddev = 0.1))
        
    def call(self, inputs):
        output1 = tf.add(tf.linalg.matmul(inputs, self.W1), self.b1)
        relu_output1 = tf.nn.relu(output1)
        logits = tf.add(tf.linalg.matmul(relu_output1, self.W2), self.b2)
        
        return logits
    
    def loss(self, logits, labels, delta_x, inputs, true_idx, false_idx):
        original_loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
        true_delta_x = tf.gather(delta_x, tf.where(true_idx))[:,0,:]
        false_delta_x = tf.gather(delta_x, tf.where(false_idx))[:,0,:]
        true_x = tf.gather(inputs, tf.where(true_idx))[:,0,:]
        false_x = tf.gather(inputs, tf.where(false_idx))[:,0,:]
        
        false_loss = []
        for i in range(false_x.shape[0]):
            false_loss.append(-self.c * tf.math.exp(-tf.norm(false_delta_x[i,:])/tf.norm(false_x[i,:])))
            
        true_loss = []
        for i in range(true_x.shape[0]):
            true_loss.append(self.d * tf.math.exp(-tf.norm(true_delta_x[i,:])/tf.norm(true_x[i,:])))
            
        loss = original_loss + self.l * tf.convert_to_tensor(false_loss + true_loss) 
        
        return tf.reduce_sum(loss)
    
    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
  
  
regu_model = regu_MLP(28*28)

def regu_train(model, regu_model, train_inputs, train_labels, delta_x):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    
    for x in range(0, train_inputs.shape[0], regu_model.batch_size):
        batch_inputs = train_inputs[x: x + regu_model.batch_size, :]
        batch_labels = train_labels[x: x + regu_model.batch_size]
        batch_delta_x = delta_x[x: x + regu_model.batch_size, :]
        
        #collect the index of correct prediction
        batch_logits = model.call(batch_inputs)
        true_idx = tf.equal(tf.argmax(batch_logits, 1), tf.argmax(batch_labels, 1))
        false_idx = tf.not_equal(tf.argmax(batch_logits, 1), tf.argmax(batch_labels, 1))
        
        
        # Optimize gradients
        with tf.GradientTape() as tape:
            predictions = regu_model.call(batch_inputs)
            loss = regu_model.loss(predictions, batch_labels, batch_delta_x, batch_inputs, true_idx, false_idx)
      
        
        gradients = tape.gradient(loss, regu_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, regu_model.trainable_variables))


    
# accuracy on the regularised model
epochs = 5
for i in range (epochs):
    regu_train(model, regu_model, all_x, y_train[:640], all_r_out)
    regu_test_acc = test(regu_model, test_all_x, y_test[:128])
    print(regu_test_acc)

print("Accuracy on the regularized test set after {} epochs steps: {}".format(epochs, regu_test_acc))



 
    

# original image
plt.imshow(tf.reshape(train_inputs[0,],[28,28]).numpy(), cmap = plt.get_cmap('gray'))
plt.show()
# after adding noise
plt.imshow(tf.reshape(x,[28,28]).numpy(), cmap = plt.get_cmap('gray'))
plt.show()

    
