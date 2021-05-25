# PIC16B Project - Colorization of Black and White Photos

## Project Proposal:

**Abstract:**
Colorization has a variety of applications in recreational and historical context. It transforms how we see and perceive photos which provides an immense value in helping us visualize and convey stories, emotion, as well as history. In this project, we will be implementing a deep neural network model on a huge data set which then will be used to colorize black and white photos. In short, what we will be planning to do in the next couple of weeks is we will scrape data using ``scrapy``, analyze and prepare the data, train them on a deep neural network deployed through TensorFlow and use ``OpenCv`` for image processing. Our ultimate goal for this project is to construct a web app that has a basic API which allows user to easily use and interact with.

**Planned Deliverables:**
The final goal is to create an interactive web app which allows users to input in any B&W photo, and output its corresponding predicted colorized version. The program is focused on reducing racial bias and accurately predicting historical photos, recreating lively colorized scenes and portraits. 
If timing is not supported for web application, our second option is to create a Python package that will accept any gray-scale photos (jpg, jpng, png, from url link, or even gif) and output a precise colorized photo.

**Resources Required:**
Since we are looking at image, Google is the best relatively random image with reduced choosing bias and data bias.
Data would be from web scraping on google image (keyword: portrait, nature, historical site,... ), convert to gray-scale using ``matplotlib`` as train/test data, test final model on B&W historical photos.

**Tools/Skills Required:**
- ``scrapy`` (Web Scrapping)
- ``tensorflow`` (Machine Learning)
- ``kera`` (Machine Learning - if applicable)
- ``OpenCv`` (Computer vision)
- ``sql`` (Database)

**Risks:**
- Need a huge and diverse data set
- Computational speed/power (CPU)

**Ethics:**
It is definitely essential to consult expert's advice on historical photos since the model may distort history as we don't have enough data from the past to train on.
Diversity in the data set would also be a real hurdle here as it is challenging to have an accurate portrayal of people’s skin color based solely on their physical appearance which leads to the unwanted racial biases in machine learning. It could happen if the data set is not diverse enough, i.e., there is too much focus on a certain race/ethnicity. One of the way to mitigate this effect is certainly to diversify our sources of data as much as we possibly can.
