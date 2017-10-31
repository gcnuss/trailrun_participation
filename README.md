# Meridian Geographics (MerGeo) Race Participation
Capstone Project for Galvanize Data Science Immersive

### Context
The Pacific Northwest is a popular place to participate in trail runs and navigation races.  There are multiple local companies who put on events in the area.  In this market, how does one company stand out amidst the pack?  As a small, local company, how can one make best use of marketing resources to increase participation and repeat customers?  I am working with local trail run company MerGeo’s participant and race data to help answer questions such as these.  The goal of this project is to better understand customer participation habits in order to improve marketing and race definition (location/timing/type) decisions, and to provide better support to customers in line with their interests where possible.

### The Data
The data for this project was pulled from MerGeo's existing MS Access Database.  Data includes participant features gathered during registration as well as numerous features describing each event.  The data includes 16,978 participations across 10,764 racers and 86 races in a roughly 2.5 year period.  The participation sparsity is 98%.  I've developed a data preparation pipeline that leverages PostgreSQL and Python to mine the data, address missing values and inconsistent data entries, and reformat features as needed to support EDA and modeling.  Mining the Access Database utilizes the Mac app "ACCDB MDB Explorer", which provides functionality to export the database as a .sql file.

### Modeling
Given the available data and the lack of explicit information on participant interests, I identified a recommender system using NMF as a good approach to identify the 'best events' without having to know all of the factors that make them the best.  I also considered a classification model for event participation, however the available data is all of the positive class (participated).  Another perspective I considered was a regression model to predict race volume, however bucketing the available data to support such a model would greatly reduce the number of observations I have to work with.
To address the cold start issue with recommending/predicting future events, instead of using unique EventID for my "item" values I created five Spark Implicit ALS models using repeatable event features (Series, Event Type, Average Mileage, Average Fee, and Venue Location by Zip Code).  These ALS models are aggregated in a Gradient Boosted Ensemble model using SkLearn.  Models were tuned and tested using a train/validate/test data split, grid search, and evaluation using percentile rank (represents how well the model ranks actually attended events in the recommendation lists - lower values indicate a good correlation, with actually attended events high on the list).
The resultant recommendation lists are used to identify which events show up the most in participants' top recommendations, and to explore trends amongst the set of users for which a given event is highly recommended.  The NMF user-feature and item-feature matrices generated during the ALS model training are also used to perform unsupervised clustering to explore what factors may be most influencing participation and what makes a 'best event'.

### Results
When evaluated, each individual ALS model as well as the Gradient Boosted Ensemble perform fairly well in terms of percentile rank.  As a point of comparison I also calculated the percentile rank score if I were to simply recommend the most popular (i.e. most highly attended) races to everyone.  The ALS and Gradient Boost models perform notably better than the popularity approach.  However, currently the individual ALS models are performing better than the ensemble due to each predicting different subsets of the data better.
The results so far indicate that there is opportunity to increase repeat participation of existing customers (brand loyalty), and opportunities for targeted marketing with the events most frequently highly recommended, and the set of people associated with a particular event in their top recommendations.  I found that race attributes vary quite a bit amongst top recommendations and trends are not clear, so more work is needed in this area before explicit conclusions can be drawn.  In addition to the modeling results, EDA also contributed to these observations.  There is a fair amount of additional EDA that can be performed to answer additional questions MerGeo has raised.

### Deployment
This project and its results will continue to be documented here on GitHub.  A project site will be added using GitHub Pages in the near future.  In addition, I am preparing a report for MerGeo and will be meeting with them in mid-November to review my work and findings.

### Future Work
**Modeling:**
* Investigate alternative ensemble methods that improve upon the individual ALS models
* Acquire more data to train additional ALS models (race start times, distance from participant home to venue, race sell out)
**EDA, Clustering:**
* Participant Geographic Trends
* Participant Registration Trends
* Participant Turnover and Factors Driving Repeat Participation
* Patterns of Participation with respect to season, time, demographics, etc
* Acquire more participant data to better inform what clusters represent
