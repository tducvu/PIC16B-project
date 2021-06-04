## Colorization of Black and White Images ([WebApp](https://colorizing-pic16b.herokuapp.com/) - In Progress)

###  [Alice Pham](https://naliph.github.io/), [Duc Vu](https://tducvu.github.io/)

<img src='demo2.gif' width=800>

For this project, we implement a convolutional neural network combined with a classifier to colorize greyscale images. As we only have access to some certain (free) resources, especially on computing power, e.g., limited access time to GPU, our model is in no way completed/perfect; it still has many flaws and needed to be trained more aggressively. For the upcoming sections, we will first describe the **file structure** of our repository. Then, we would get into the **prerequisites** and **installation** of the colorizing app. Lastly, we will discuss some of the limitation and challenges that we face throughout this project.

### Repository Structure

|Folder            | Description |
|------------------| ----------- |
|`Webapp/`         | contains all the `.py` files which are needed to deploy the `streamlit` app|
|`models/`         | pre-trained models which would be loaded to the app to colorize images|
|`colorizer/`      | Jupyter Notebook [(colorizer.ipynb)](https://github.com/tducvu/PIC16B-project/blob/main/Colorizer/colorizer.ipynb) used to implement and train the model|
|`Data/`           | data sources used to train the model on|


**Some additional files needed to deploy heroku**:

| File              | Description |
|-------------------|-------------|
|`runtime.txt`      | the required version of **Python**|
|`requirements.txt` | the required and specified version packages needed to run [colorizeApp.py](https://github.com/tducvu/PIC16B-project/blob/main/Webapp/colorizeApp.py)|
|`setup.sh`         | setup port to deploy **heroku** webapp|
|`Procfile`         | **heroku** special file indicate how to set up and run the source code|


### Installation

1. **Prerequisites**
-  Since this app needs to be ran on some specific versions of some packages, we recommend to install [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) to run the app on a virtual environment which reduces the risk of breaking your base environment.
- GPU or any platforms such as [Google Colab](https://research.google.com/colaboratory/) and [FloydHub](https://www.floydhub.com/) that give you free access to limited GPU computing power (only applicable if you want to train your own model on [colorizer.ipynb](https://github.com/tducvu/PIC16B-project/blob/main/Colorizer/colorizer.ipynb).

2. **Getting Started**
- After installing **conda**, run the following commands in a terminal to create a new environment

```bash
conda create -n colorizer python=3.6
conda activate colorizer
conda install --file requirements.txt
```

- Next, let's clone this repository and change our current working directory.

```bash
git clone https://github.com/tducvu/PIC16B-project.git
cd PIC16B-project
```

- Now, to simply test out our app, just execute the following commands

```bash
cd Webapp
streamlit run colorizeApp.py
```

- Otherwise, to build and train your own model, please take a look and follow the instruction in [colorizer.ipynb](https://github.com/tducvu/PIC16B-project/blob/main/Colorizer/colorizer.ipynb).

3. **Note**: 

Our local app takes quite a noticeable amount of time to run which is due to the loading of a pre-trained model and a rather heavy and powerful classifier - [InceptionResNetV2](https://keras.io/api/applications/inceptionresnetv2/).

### Limitation & Improvement
Since we can only train at most on 10,000 images and a relatively small number of epochs due to the lack of a strong processing core, the colorized image does not appear to be very polished, convincing and as sharp as we want it to be (nevertheless, it can make your B&W pictures looks very artistic-ish :D ). Undoubtedly, with more time and computing power, we can train our model on more images and epochs which can potentially increases the sensitivity and "aesthetic" of the model.

Moreover, because of the heavy model and long runtime to colorize, the option to run this app on **heroku** is still currently in progress (connection timeout error). In addition, this is our first time using **heroku**, and we just figured out recently that it's not a suitable platform to deploy machine learning models as they tend to be very resource-hungry/intensive (which ours is). Thus, more research need to be done here to find a more appropriate host for the webapp.

### Source
The algorithm implemented in [colorizer.ipynb](https://github.com/tducvu/PIC16B-project/blob/main/Colorizer/colorizer.ipynb) is from [Emil's](https://www.emilwallner.com/) blog post on colorization which is posted on [medium](https://emilwallner.medium.com/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d). We're very grateful for his amazing and eye-opening work. Please visit the site for more detailed and thorough explanation of the colorizing process.
