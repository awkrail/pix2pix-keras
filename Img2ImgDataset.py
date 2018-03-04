from PIL import Image
import numpy as np
import os


class FlightDataset(object):
    def __init__(self, input_path, output_path, data_range):
        """
        dataset loader and obtain dataset.
        load dataset train and valid.
        params:
            self.in_array => input dataset
            self.out_array => output dataset
        """
        print("loading dataset...")
        self.in_array = np.zeros((data_range[1] - data_range[0], 224, 224, 1))        
        self.out_array = np.zeros((data_range[1] - data_range[0], 224, 224, 3))

        for i in range(data_range[0], data_range[1]):
            if self.check_img_path_in_paths(i, input_path, output_path):
                continue
            label = Image.open(input_path+"image_%04d.jpg"%i)
            img = Image.open(output_path+"image_%04d.jpg"%i)
            # resize
            resized_img = img.resize((224, 224), Image.NEAREST)
            resized_label = label.resize((224, 224), Image.NEAREST)

            resized_img = np.asarray(resized_img).astype("f") / 128.0 - 1.0
            resized_label = np.asarray(resized_label)
            resized_label = resized_label.reshape(224, 224, 1)
            self.in_array[i] = resized_label
            self.out_array[i] = resized_img
    
    @staticmethod
    def check_img_path_in_paths(i, in_path, out_path):
        if not "image_%04d.jpg" in os.listdir(in_path) and not "image_%04d.jpg" in os.listdir(out_path):
            return True
        return False
