# Author - Kavya Jayaramaiah
# idmid: iz81eniq

import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        # TODO: implement constructor

        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.epoch_no = 0


        self.file_path = file_path
        self.file_list = self.getFilelist()
        self.label_path = label_path
        self.labels_dict = self.readJson()


        arr_seq = np.arange(len(self.labels_dict)) #its like the sequence 
        if self.shuffle:
            np.random.shuffle(arr_seq)# randomly placed the object each eopch, avoids over fitting 
        self.arr_seq = arr_seq# stores the random number
        self.end_position = 0# acts like a pointer to track the batch where extaclly it is the points 


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # TODO: implement next method
        # TODO: Find Random method
        # TODO: Find circular for loop

        
        batch_images = []
        batch_labels = []

        # Create the new epoch 

        if self.shuffle and (self.epoch_no != (self.end_position // len(self.arr_seq))):
            print('Inside the shuffle loop')
            self.epoch_no = self.current_epoch()
            arr_seq = np.arange(len(self.labels_dict))
            np.random.shuffle(arr_seq)
            self.arr_seq = arr_seq

        for i in range(self.end_position, self.end_position + self.batch_size, 1):

            temp_images = self.readSingleImg(self.file_list[self.arr_seq[i % len(self.arr_seq)]])
            temp_label = self.labels_dict[self.arr_seq[i % len(self.arr_seq)]]

            if self.mirroring or self.rotation:
                temp_images = self.augment(temp_images)

            batch_images.append(temp_images)
            batch_labels.append(temp_label)

        self.end_position = self.end_position + self.batch_size
        batch_images = np.array(batch_images)

        return batch_images, batch_labels


    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function
         
         
        img_aug = img.copy()
        if self.mirroring:

            if self.mirroring and np.random.rand() < 0.5:
                img_aug = np.flip(img_aug, axis=1)  # returns the mirror image of an array

        if self.rotation:

            if self.rotation and np.random.rand() < 0.5:
                degree = np.random.choice([1, 2, 3])  # Randomly returns one of the specified values
                img_aug = np.rot90(img_aug, k=degree)  # Number of times the array is rotated by 90 degrees

        return img_aug

    def current_epoch(self):
        # return the current epoch number
        epoch_val = (self.end_position - 1) // len(self.arr_seq)
        if epoch_val == -1:
            epoch_val = 0
        return epoch_val

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function

        return self.class_dict[x]

    def show(self):

        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method
        current_images, current_label = self.next()
        column_size = 3
        row_size = (self.batch_size + column_size - 1) // column_size

    
        fig = plt.figure(figsize=(column_size * 4, row_size * 4))

        for i in range(self.batch_size):
            sb = fig.add_subplot(row_size, column_size, i + 1)
            sb.imshow(current_images[i], aspect='auto')
            sb.axis('off')
            sb.set_title(self.class_name(current_label[i]))

        # plt.tight_layout()
        plt.show()

    def getFilelist(self):

        file_dir = []
        dir_data = os.listdir(self.file_path)
        dir_data = sorted(dir_data, key=lambda x: int(os.path.splitext(x)[0]))

        
        for filename in dir_data:
            
            if os.path.isfile(os.path.join(self.file_path, filename)):
                file_dir.append(os.path.join(self.file_path, filename))

        return file_dir

    def readJson(self):

        json_file = open(self.label_path)
        data = json.load(json_file)
        json_file.close()
        val = [[int(x) for x in list(data.keys())], list(data.values())]
        new_dict = dict(zip(val[0], val[1]))
        return new_dict

    def readImgdata(self):
        imgdata = []
        file_list = self.getFilelist()

        for i in range(len(file_list)):
            tmp_data = np.load(file_list[i])
            tmp_data = resize(tmp_data, self.image_size)
            imgdata.append(tmp_data)

        return imgdata

    def readSingleImg(self, file_name):
        imgdata_x = np.load(file_name)
        imgdata_x = resize(imgdata_x, self.image_size)
        return imgdata_x