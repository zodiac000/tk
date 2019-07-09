import tensorflow as tf
import numpy as np
from dataset import get_ds, get_data_list
from model import create_model


pred_dict = "pred_dict.csv"
saved_dict = 'saved_dict.csv'
batch_size = 1000
width = 1280
height = 1024

_, ds_test = get_ds(saved_dict, 1, batch_size)
ds_test = ds_test.take(1)

model = create_model()
model.summary()
model.load_weights('mymodel.h5')

all_image_paths, all_x, all_y = get_data_list(saved_dict)
predictions = [model.predict(x, steps=1) for x in ds_test]


def save_xy():
    with open(pred_dict, "w") as f:
        for i, pred in enumerate(predictions[0]):
            f.write(all_image_paths[i] + "," + str(int(pred[0] * width)) + "," + str(int(pred[1] * height)) + "\n")
    print("write {} predictions to {}".format(batch_size, pred_dict))

for index, pred in enumerate(predictions[0]):
    print("{}: predicted: ({}, {})\tlabel: ({}, {})".format(all_image_paths[index],
                                                            int(pred[0] * width), 
                                                            int(pred[1] * height), 
                                                            all_x[index], all_y[index]))
save_xy()
