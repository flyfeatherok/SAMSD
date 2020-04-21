import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os, random
import numpy as np
from glob import glob

class ImageData:

    def __init__(self, data_path, img_shape=(64,64,1), augment_flag=False, data_type='None', img_type='jpg', pad_flag=False, label_size=8):
        self.data_path = data_path
        self.data_type = data_type
        self.img_shape = img_shape
        self.img_h = img_shape[0]
        self.img_w = img_shape[1]
        self.channels = img_shape[2]
        self.augment_flag = augment_flag
        self.img_type = img_type
        self.pad_flag = pad_flag
        self.label_size = label_size
        if self.data_type == 'CelebA':
            self.CelebaA = CelebaA_Data(self.data_path, self.label_size)
            self.train_dataset, self.train_label = self.CelebaA.CelebaA_Data_Label()
        else:
            self.train_dataset = glob(os.path.join(os.getcwd(), self.data_path, '*.'+img_type))
            self.train_label = np.zeros(len(self.train_dataset))

    def image_processing(self, filename, label):
        x = tf.read_file(filename)
        if self.img_type == 'jpg':
            x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        if self.img_type == 'bmp':
            x_decode = tf.image.decode_bmp(x)
            if self.channels == 1 :
                x_decode = tf.image.rgb_to_grayscale(x_decode)
        img = tf.image.resize_images(x_decode, [self.img_h, self.img_w])
        img = tf.reshape(img, [self.img_h, self.img_w, self.channels])
        if self.pad_flag==True:
            img = tf.pad(255-img, [[3, 4], [4, 5], [0,0]])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag :
            p = random.random()
            if p > 0.5:
                img = augmentation(img)
        
        return img, label

class CelebaA_Data:

    def __init__(self, data_path, label_size) :
        self.data_path = data_path
        self.lines = open(os.path.join(data_path, 'list_attr_celeba.txt'), 'r').readlines()
        self.label_size = label_size
        self.labels = ['Male','Pale_Skin','Black_Hair','Blond_Hair',
                        'Brown_Hair','Gray_Hair','Eyeglasses','Young',
                        'Smiling','Mouth_Slightly_Open','Bangs','Bald']
        self.Label_Mode={}
        for i in range(len(self.labels)):
            self.Label_Mode[self.labels[i]]=i
        all_attr_names = self.lines[1].split()
        self.attr2idx = {}
        self.idx2attr = {}
        for i, attr_name in enumerate(all_attr_names) :
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        self.lines = self.lines[2:]
        self.idx = []
        for l in self.labels:
            self.idx.append(self.attr2idx[l])
    
    def CelebaA_Data_Label(self):
        data = []
        label=[0.0]*self.label_size
        labels=[]
        for line in self.lines:
            label=[0.0]*self.label_size
            split = line.split()
            filename = os.path.join(self.data_path, 'celeba',split[0])
            att_value = split[1:]
            data.append(filename)
            for i in range(self.label_size):
                if att_value[self.idx[i]] == '1':
                    label[self.Label_Mode[self.labels[i]]] = 1
            labels.append(label)
        return data, labels
    
    def CreateLabel(self, modes):
        label=[0.0]*self.label_size
        for mode in modes:
            label[self.Label_Mode[mode]] = 1
        return np.float32(label)
    
def one_hot(batch_size, mask_size, location):
    l = tf.constant([location])
    m = tf.one_hot(l,mask_size,1.,0.)
    m = tf.tile(m,[batch_size,1])
    return m
    
def load_test_data(image_path, size_h=256, size_w=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size_h, size_w])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def augmentation(image):
    seed = random.randint(0, 2 ** 31 - 1)
    image = tf.image.random_flip_left_right(image, seed=seed)
#    image = tf.image.random_brightness(image,max_delta=0.2)
#    image = tf.image.random_contrast(image, 0.5, 1.5)
#    image = tf.clip_by_value(image,-1.,1.)
#    image = tf.image.random_saturation(image, 0, 0.3)
    return image

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def summary(tensor_collection, summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']):
    """
    usage:

    1. summary(tensor)

    2. summary([tensor_a, tensor_b])

    3. summary({tensor_a: 'a', tensor_b: 'b})
    """

    def _summary(tensor, name, summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']):
        """ Attach a lot of summaries to a Tensor. """

        if name is None:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            name = re.sub('%s_[0-9]*/' % 'tower', '', tensor.name)
            name = re.sub(':', '-', name)

        with tf.name_scope('summary_' + name):
            summaries = []
            if len(tensor.shape) == 0:
                summaries.append(tf.summary.scalar(name, tensor))
            else:
                if 'mean' in summary_type:
                    mean = tf.reduce_mean(tensor)
                    summaries.append(tf.summary.scalar(name + '/mean', mean))
                if 'stddev' in summary_type:
                    mean = tf.reduce_mean(tensor)
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
                    summaries.append(tf.summary.scalar(name + '/stddev', stddev))
                if 'max' in summary_type:
                    summaries.append(tf.summary.scalar(name + '/max', tf.reduce_max(tensor)))
                if 'min' in summary_type:
                    summaries.append(tf.summary.scalar(name + '/min', tf.reduce_min(tensor)))
                if 'sparsity' in summary_type:
                    summaries.append(tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor)))
                if 'histogram' in summary_type:
                    summaries.append(tf.summary.histogram(name, tensor))
            return tf.summary.merge(summaries)

    if not isinstance(tensor_collection, (list, tuple, dict)):
        tensor_collection = [tensor_collection]
    with tf.name_scope('summaries'):
        summaries = []
        if isinstance(tensor_collection, (list, tuple)):
            for tensor in tensor_collection:
                summaries.append(_summary(tensor, None, summary_type))
        else:
            for tensor, name in tensor_collection.items():
                summaries.append(_summary(tensor, name, summary_type))
        return tf.summary.merge(summaries)
