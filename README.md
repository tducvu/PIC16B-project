# PIC16B-project

Tic Tac Toe <br>
O    O     O <br>
.    X     . <br>
X    .     X <br>


## Project Proposal:

**Abstract: (2-4 sentences)**
Colorization has a variety of applications in recreational and historical context. It transforms how we see and perceive photos which provides an immense value in helping us visualize stories, emotion, as well as history. In this project, we will be implementing a deep neural network model on a huge data set which then will be used to colorize black and white photos. In short, what we will be doing in the next couple of weeks is we will scrape data using Scrapy, analyze and prepare the data, feed it to a deep neural network deployed through TensorFlow and use OpenCv for image processing. Our ultimate goal for this project is to construct a web app that has a basic API which allows user to easily use and interact with.

**Planned Deliverables:**
The final goal is to create an interactive web app which allows users to input in any B&W photo, and output its corresponding predicted colorized version. The program is focused on reducing racial bias and accurately predicting historical photos, recreating lively colorized scenes and portraits. 
If timing is not supported for web application, our second option is to create a Python package that will accept any gray-scale photos (jpg, jpng, png, from url link, or even gif) and output a precise colorized photo.

**Resources Required:**
Since we are looking at image, Google is the best relatively random image with reduced choosing bias and data bias.
Data would be from web scraping on google image (keyword: portrait, nature, historical site,... ), convert to gray-scale using matplotlib as train/test data, test final model B&W historical photos

**Tools/Skills Required:**
- Scrapy
- Tensorflow
- Kera (if applicable)
- OpenCv (Computer vision)
- SQL
- ....

**Risks:**
- Need a huge and diverse data set
- Computational speed/power (cpu)
- ...

**Ethics:**
Need expert advice on old photos since this may distorts historical truths.
Diversity in the data set would again be a real hurdle here as it’s challenging to have an accurate portrayal of people’s skin color based solely on their physical appearance, etc
Racial bias could happen if the resources data set is not diverse enough, too focus on training a certain race/ethnicity for example. Need to check the database. 


**Tentative timeline**
Week 3: Create repository <br>
Week 3: Web scraping for train/test data set of 1,000,000 <br>
Week 4: Data Exploratory (understand color distribution, understanding Tensorflow and Open CV)<br>
Week 5: Model selection based on exploratory hypothesis.<br>
Week 6: Create a tested model (at least have an out image - no need to be accurate about color yet)<br>
Week 7: Turn in draft model<br>
Week 8: Explore either Web app or python package, refined model<br>
Week 9: tbd <br>
Week 10: Final Project due

