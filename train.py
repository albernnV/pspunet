import numpy as np
import tensorflow as tf
from data_loader import convert_class, data_load
from display import create_mask, show_predictions
from IPython.display import clear_output
from pspunet import pspunet
import matplotlib.pyplot as plt
import datetime
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


IMG_WIDTH = 480
IMG_HEIGHT = 272
n_classes = 10
gpus = tf.config.experimental.list_physical_devices('GPU')
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
print(gpus)

if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
       gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16000)])
    except RuntimeError as e:
        print(e)
 

        
model = pspunet((IMG_HEIGHT, IMG_WIDTH,3), n_classes)

train_dataset, validation_dataset, test_dataset = data_load()

optimizer = tf.keras.optimizers.Adam(1e-4)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

validation_loss = tf.keras.metrics.Mean(name='validation_loss')
validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='validation_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

train_mIoU =tf.keras.metrics.MeanIoU(num_classes = n_classes, name = "train_mIoU")
validation_mIoU =tf.keras.metrics.MeanIoU(num_classes = n_classes, name = "validation_mIoU")
test_mIoU =tf.keras.metrics.MeanIoU(num_classes = n_classes, name = "test_mIoU")

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '/pspunet/'
validation_log_dir = '/pspunet/' 
test_log_dir = '/pspunet/'
#train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#validation_summary_writer = tf.summary.create_file_writer(validation_log_dir)
#test_summary_writer = tf.summary.create_file_writer(test_log_dir)

@tf.function
def train_step(images, label):
    
    with tf.GradientTape() as tape:
        pred_img = model(images)        
        loss =  loss_object(label, pred_img)

    gradients_of_model = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(label, pred_img)

@tf.function
def validation_step(images, label):
    pred_img = model(images)
    loss =  loss_object(label, pred_img)
    
    #pred_mask = tf.argmax(pred_img, axis=-1)
    #pred_mask = pred_mask[..., tf.newaxis]
    
    #test_mIoU(label, pred_mask)
    validation_loss(loss)
    validation_accuracy(label, pred_img)
 
@tf.function
def test_step(images, label):
    pred_img = model(images)
    loss =  loss_object(label, pred_img)
    
    #pred_mask = tf.argmax(pred_img, axis=-1)
    #pred_mask = pred_mask[..., tf.newaxis]
    
    #test_mIoU(label, pred_mask)
    test_loss(loss)
    test_accuracy(label, pred_img)

def train(train_dataset, test_dataset, epochs, batch_size):
    epoch_train_losses = []
    epoch_val_losses = []
    with tf.device('/device:GPU:0'):
        for epoch in range(epochs):
            total_epoch_train_loss = 0
            total_epoch_val_loss = 0
          
            if epoch >=10:
                optimizer = tf.keras.optimizers.Adam(1e-5)

          
            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()    
          
            test_loss.reset_states()
            test_accuracy.reset_states()
                  
            count_img = 0
            batch_time = time.time()
          
            for image_batch, label_batch in train_dataset.batch(batch_size):
                count_img += batch_size
                label_batch = convert_class(label_batch.numpy())
              
                if tf.random.uniform(()) > 0.5:
                    image_batch = tf.image.flip_left_right(image_batch)
                    label_batch = tf.image.flip_left_right(label_batch)

                    image_batch = tf.image.random_brightness(image_batch, 0.3)
              
                train_step(image_batch, label_batch)
              
                if count_img % 8 == 0:
                    total_epoch_train_loss += train_loss.result()

                    clear_output(wait=True)
                    show_predictions(image_batch[:3], label_batch[:3], model)      
                    print('epoch {}, step {}, train_acc {}, loss {} , time {}'.format(epoch+1,
                                                                          count_img,
                                                                          train_accuracy.result()*100,
                                                                          train_loss.result(),
                                                                          time.time()- batch_time))
                    train_loss.reset_states()
                    train_accuracy.reset_states()    
                  
                    batch_time = time.time()
                  
            epoch_train_losses.append(total_epoch_train_loss/(count_img/batch_size))
            count_img = 0
            batch_time = time.time()

            for image_batch, label_batch in validation_dataset.batch(batch_size):
                count_img += batch_size
                label_batch = convert_class(label_batch.numpy())
              
                validation_step(image_batch, label_batch)
              
              
                if count_img % 8 == 0:
                    total_epoch_val_loss += validation_loss.result()
                    clear_output(wait=True)
                    show_predictions(image_batch[:3], label_batch[:3], model)
                    print('epoch {}, step {}, test_acc {}, loss {} , time {}'.format(epoch+1,
                                                                          count_img,
                                                                          validation_accuracy.result()*100,
                                                                          validation_loss.result(),
                                                                          time.time()- batch_time))
                    batch_time = time.time()
                  
            epoch_val_losses.append(total_epoch_val_loss/(count_img/batch_size))
                  
            clear_output(wait=True)

            for image_batch, label_batch in validation_dataset.take(3).batch(3):
                label_batch = convert_class(label_batch.numpy())
                show_predictions(image_batch[:3], label_batch[:3], model)        

            print ('Time for epoch {}  is {} sec'.format(epoch + 1, round(time.time()-start),3))

            print ('train_acc {}, loss {} , validation_acc {}, loss {}'.format(train_accuracy.result()*100,
                                                                                  train_loss.result(),
                                                                                  validation_accuracy.result()*100,
                                                                                  validation_loss.result()
                                                                                  ))
          
            path = "/pspunet/" + str(validation_loss.result().numpy())+"_epoch_"+str(epoch+1)+".h5" 
            model.save(path)
              
            #with train_summary_writer.as_default():
             #   tf.summary.scalar('loss', train_loss.result(), step=epoch+1)
              #  tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch+1)             
               # tf.summary.scalar('mIoU', train_mIoU.result(), step=epoch)    
                  
            #with validation_summary_writer.as_default():
             #   tf.summary.scalar('loss', validation_loss.result(), step=epoch+1)
              #  tf.summary.scalar('accuracy', validation_accuracy.result(), step=epoch+1)    
               # tf.summary.scalar('mIoU', validation_mIoU.result(), step=epoch)     

        plt.plot(range(1,len(epoch_train_losses)+1),epoch_train_losses)
        plt.plot(range(1, len(epoch_val_losses)+1), epoch_val_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
      

        print('Finished Training')


train(train_dataset, validation_dataset, 1, 8)

model.summary()


count_img = 0
batch_time = time.time()

for image_batch, label_batch in test_dataset.batch(8):
    count_img += 8
    label_batch = convert_class(label_batch.numpy())
    
    test_step(image_batch, label_batch)
            
    if count_img % 8 == 0:
        clear_output(wait=True)
        show_predictions(image_batch[:3], label_batch[:3], model)
        print('epoch {}, step {}, test_acc {}, loss {} ,mIoU {}, time {}'.format(1,
                                                                count_img,
                                                                test_accuracy.result()*100,
                                                                test_loss.result(),
                                                                test_mIoU.result(),
                                                                time.time()- batch_time))
        batch_time = time.time()