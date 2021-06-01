# PIC16B Project - Colorization of Black and White Photos

#### The main requirement file/ folder for the project:

| File/Folder      | Description |
|---------------- | ----------- |
| models           | Pretrained models to run colorization|
|support.py        | contains `create_inception_embedding`, and `load_pretrained_model()` function, support for **serving.py** file|
| serving.py       | contains `load_model()`, `evaluate_input()`, and `_data_preprocessing` function, serves main **colorizeApp.py** file|
| requirements.txt | The required libraries with version to run file colorizeApp.py|
| .streamlit      | folder that contains hidden config.toml of streamlit, the config can be show using `$streamlit config show`|
| Colorizer      | contain **Test** folder with tester B&W images for the app |




#### Extra file for styling:

| File/Folder | Description |
|-------------|-------------|
|style.scss   | added scss style for streamlit app|
|load_css.py  | function `local_scss()` load scss to markdown in `st.markdown()`, to be called in colorizeApp.py|


#### File to assist deploying streamlit app to heroku:

| File/Folder | Description |
|-------------|-------------|
| runtime.txt       | The required version of Python = 3.6.13|
| setup.sh | setup port for heroku using `config.toml` from **.streamlit** folder|
| Procfile | heroku special file indicate `web: sh setup.sh && streamlit run colorizeApp.py`|
| Aptfile | contains support file for openCV library need for deploy|
| .slugignore | reduce app size by ignore unecessary file in repository|







## Tutorial for colorization:
First of all, fork this repository :) <br>

Since we use an older version of Python and of some libraries, you will need to create another environment. Suggesting download Conda starting from the very beginning ([conda docs](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html))

After having conda in path, let's get start!

1. Create new environment that host these libraries with Python version 3.6, activate it and install packages as requirements.txt indicate:
``` 
$ conda create -n colorizer python 3.6
$ conda activate colorizer
$ conda install --file requirements.txt
```

2. Go to the repository folder, and run the streamlit command:
```
$ streamlit run colorizeApp.py

```


3. Done, you can play around with the local app, add some B&W picture and see some color!
The streamlit contains 2 method of colorization: upload your own B&W pictures, or use the 10 tester images that we provide in the app. Any case, the localhost app may run slow due to loading heavy pretrained model.


## Limitation
Since we can only train at most on 10,000 images and little epochs number due to lack of a strong processing core, the colorization images does not work perfectly and not as sharp as we want it to be (but, it can make your B&W pictures looks very artistic-ish :D ). 

The **colorizer** folder contains the colorizer.ipynb file, which include explanation and our code with convolutional neural network training process.
There is a choice to retrain the model with a stronger GPU if you want to for a better colorization model.


Moreover, currently, due to the heavy model and long runtime to colorize, the option to deploy to heroku app is unavailable and it's not the best choice to upload a data science project. In the future, we can research more about methods to deploy the web app with a better colorization model.
