To go about partitioning the images we would run
```
python3 partition_training_image_full_color.py
```
Next we create a csv with the name of the files by running
```
python3 make_train.py
```
After we can go about training one of our models, this being the logistic regression model by running

```
python3 train_lr.py
```

To train the linear support vector machine model we run

```
python3 train_svm.py
```

Next we go into the gpu_model/ folder
