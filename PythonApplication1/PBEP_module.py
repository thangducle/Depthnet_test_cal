import tensorflow as tf
import keras

# tf.config.list_physical_devices('GPU')

# PBEP module
class PBEP(tf.keras.Model):
  def __init__(self, filters):
    super(PBEP, self).__init__(name='')
    

    self.conv1a = tf.keras.layers.Conv2D(filters*2, (1, 1))  # A projection layer
    self.bn1a = tf.keras.layers.BatchNormalization()  # A batch normalization layer
    self.conv1a = tf.keras.layers.Conv2D(filters, (1, 1))  # An expansion layer

    self.dwconv1a = tf.keras.layers.SeparableConv2D(filters, (3,3), depth_multiplier = 1, padding="same")  # A depth-wise convolution layer

    self.conv1b = tf.keras.layers.Conv2D(filters, (1, 1))  # A projection layer


  def call(self, input_tensor, training=False):
    x = self.conv1a(input_tensor)
    x = self.bn1a(x, training=training)
    x = tf.nn.relu(x)
    x = self.dwconv1a(x)
    x = self.conv1b(x)
    x += input_tensor


    return x

PBEP_block = PBEP(3)


# [1,18,58,3] vs. [1,20,60,3] 

_ = PBEP_block(tf.zeros([1, 20, 60, 3]))

print("check vars no \n")
print(len(PBEP_block.variables))
PBEP_block.summary()
