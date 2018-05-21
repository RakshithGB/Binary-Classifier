# Binary_Classifier

A simple Cat-Dog Classifier built on `ResNet-50` using `Keras 2.1.5` with `Tensorflow (1.4) `backend on the Kaggle Dataset ([link](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765)).

## Prerequisites
1. Make sure the dataset from this [link](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765) is extracted inside the project folder.
2. `Python 3+`
3. Install Tensorflow:
    
    ```
    $ pip install tensorflow
    ```
4. Install Keras (2.1.5+):
    
    ```
    $ pip install keras
    ```

5. Install h5py:
    
    ```
    $ pip install h5py
    ```
6. Install PIL (Pillow):

    ```
    $ pip install Pillow
    ```
    
Detailed guides to install these libraries can be found [here](https://github.com/RakshithGB/Installation-Guides) for `MacOS` and `Ubuntu`.

## Testing
To test a single image for prediction open terminal and run ( `Ubuntu/macOS` recomended ):

```
$ python predict.py
```
By default the above command will classify the first image in `test_images` directory.

To pass custom images for testing:

```
$ python predict.py --image <image name>
```
Fill the <image name> with the custom image name. Even custom model can be passed as flag for testing:

```
$ python predict.py --image <image name> --model <model name>
```
All these flags are given default value so it will execute the script directly without specifying them explicitly.

## Training
The dataset in the mentioned [link](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765) will download the folder `kagglecatsanddogs_3367a` with a subfolder named `PetImages`.

Copy the subfolders of `PetImages` into `Data\Cat` and `Data\Dog` accordingly.

The obtained dataset has too many corrupt images. We first need to delete them. This is done by simply running the script ( works only on `Ubuntu/macOS` ):

```
$ python clean.py
```
We can then start the training:

```
$ python train.py
```
Parameters like `resolution`, `batch_size` etc can be changed in the script.

## Model Compression (Optional)
In order to compress the model with the tradeoff of accuracy, the following steps can be done (High RAM, Good CPU (8 core) and Good GPU recommened):

1.`$ git clone https://github.com/DwangoMediaVillage/keras_compressor.git`
2. `$ cd ./keras_compressor`
3. `pip install .`

This tool is used to compress the model to reduce disk usage.

Once training is done the script would generate a file named `resnet50_best.h5`, we can compress this file using `terminal` command:

To control the accuracy reduction flag can be passed like this:

```
$ keras-compressor.py --error 0.001 resnet50_best.h5 compressed.h5
```

Which would generate a new compressed file name `compressed.h5`.

The compression takes a very long time since it looks at every layer one by one and compresses it.

## Tensorboard
To visualise the network graph and the training loss, run in `terminal`:

```
$ tensorboard --logdir=<full path to log dir>
```
replace `<full path to log dir>` with your full path.

This will generate an IP Address, navigate to this IP Address using a web browser.

## Results
These results were obtained after deleting the corrupt images using a batch size of `16` and split ratio of `0.3` (training and testing split). It was trained for `12` epochs but the loss stopped improving at the `4th` epoch.

| Category | % |
| --- | --- |
| Training Loss | 8.5% |
| Training Accuracy | 96.6% |
| Validation Loss |  9.3%|
| Validation Accuracy |  96.55%|
| Test Weights Accuracy |  95.97%|

 

