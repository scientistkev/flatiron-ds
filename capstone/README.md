![Freight train in action](https://ithinkbigger.com/wp-content/uploads/2019/03/freight-logistics-708x440.jpg)

# Predicting Truck Freight Transport in the Heartland: A Data Science Project from Model to Deployment
## Capstone Project: Flatiron School Data Science Bootcamp

### Conducted by: Kevin McPherson | Submitted on July 23, 2020 | Presented on July 24, 2020

## Abstract

Due to it's central location in the United States, Kansas City, Missouri is
a prime location for not only distribution centers but the freight car industry. It is estimated that because of its location, 85 percent of the US population could be reached within two days of shipping and freight logistics. It is predicted that between 2016 and 2027, the shipping by truck industry will grow by 27 percent; that's growth from an already staggering $796.7 billion total revenue industry (Statista, 2018). As such, the industry demands a smarter solution to price trips before they happen, so that less operational costs can be spent on coordination of resources and more money can be put to the salaries of drivers and truck maintenance. Thus, in this capstone project, I will introduce a machine learning model that predicts shipping prices on a smaller attainable market: that of the $12 billion per year vehicle shipping services. My goal is to present both supervised (multiple linear regression), unsupervised (clustering), and ensemble (random forest regressor) learning models and deploy one of them to a requestable API used the FastAPI and Cortex technologies, which have been recently developed. 

## Some Notes Before We Proceed

* This work was done simulataneously as a work project for my career at a small startup in Kansas City called Bellwethr for another small startup in  the freight space called SuperDispatch. 
* In the project, you may see some acronyms like sd (SuperDispatch) to signify various variables and file assignments
* The data will be hidden from plain view and added to a .gitignore folder as it is proprietary and I did not previously go through the correct channels to have it shown on this repository

## Project Outline (What's Here)

The project consists of three different levels or directories:

**1. The API** <br>
This is where the API, which is tested locally via FastAPI and deployed on a staging cluster via Cortex, is stored. It is part of a larger repo that is maintained at my current job (Bellwethr), which is private and only viewable to me and the rest of the team. During my presentation, I intend to take whoever is interested through this private repo, and it is added as a submodule here. 

- The API also formalizes a process for the team to deploy a larger MLOps strategy. It utilizes the `poetry` installation module as well as writes a variety of config files that interact with AWS S3 resources that store the model and toml files that orchestrate various environment variables that work in symphony. More on this another time. 

- The idea for the API is simple: the client (a small startup called SuperDispatch) wants to use this API to quickly estimate transportation costs for customers who use their CarrierTMS or ShipperTMS software.The CarrierTMS software gives freight companies the opportunity to price out their various shipping needs while the ShipperTMS software allows merchandise owners (car dealerships, independent owners) the ability to estimate logistics costs. Currently, according to another company called Backlot Cars, this process is time-consuming and overly engineered, running through 4 different teams and spreading geographically, making pricing options costly, temporally long, and just plain headache-y. 

**2. The Notebooks** <br>
This is where I conducted all my experiments and constructed a baseline model. In theory, data science should be conducted as not knowing the 

**3. The Data** <br>

**4. The Images** <br>
This is where all the images output from the notebook are maintained. Some have code outputs like `distplot.savefig('image.png)` and others were screenshot and added to the directory.

## Materials (i.e., Libraries and Technologies) and Methods Used

The following methods and materials were used in this project, including a brief list of the 