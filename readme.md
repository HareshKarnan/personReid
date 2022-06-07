### Contrastive Descriptor learning for person reidentification using Triplet Loss

download the RAID dataset from this link : https://utexas.box.com/s/5xry50wd2nopyza5xdjvvdu5fvjjm0a1

## Run Training

To start training, first download the RAID dataset, then run the following command to start help to train the network : 

``python3 train_triplet.py -h``

make sure you use the right arguments that point to the data directory correctly. 

visualize the training using tensorboard. 

```tensorboard --logdir=lightning_logs/```

## Run Inference

The file `run_inference.py` has an example script of reading an image using opencv, converting it to pytorch tensor and running inference on the trained network by loading the model checkpoint file. 