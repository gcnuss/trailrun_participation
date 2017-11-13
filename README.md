# Meridian Geographics (MerGeo) Race Participation
Capstone Project for Galvanize Data Science Immersive

### Context
The Pacific Northwest is a popular place to participate in trail runs and navigation races.  There are multiple local companies who put on events in the area.  In this market, how does one company stand out amidst the pack?  As a small, local company, how can one make best use of marketing resources to increase participation and repeat customers?  I am working with local trail run company MerGeo’s participant and race data to help answer questions such as these.  The goal of this project is to better understand customer participation habits in order to improve marketing and race definition (location/timing/type) decisions, and to provide better support to customers in line with their interests where possible.

### The Data
* MerGeo MS Access Database
* Data includes participant and event features
* 16,978 participations across 10,764 racers and 86 races in a roughly 2.5 year period
* Utilized Mac app "ACCDB MDB Explorer" to export the MS Access database as a .sql file
* Preparation pipeline leveraging PostgreSQL and Python to address data cleaning and feature engineering

### Modeling
**Selected Model Approach**
* Implicit Recommender System using NMF to identify the 'best events'
  * Address cold start issue with recommending/predicting future events by using repeatable event features (Series, Event Type, Avg Mileage, Avg Fee, Venue)
* Five Spark Implicit ALS models aggregated in a Gradient Boosted Ensemble
* Models tuned and tested using a train/validate/test data split, grid search, and evaluation using percentile rank
* Percentile Rank represents how well the model ranks actually attended events in the recommendation lists - lower values indicate a good correlation, with actually attended events high on the list


**Other Approaches Considered:**
* Classification model for event participation, however the available data is all of the positive class (participated)
* Regression model to predict race volume, however bucketing the available data for this results in very few observations to work with

### Results
**Technical Results**
* Perecentile Rank scores performed fairly well, and notably better than a simple popularity-based recommender
* Resultant event recommendation lists used to:
  * Identify which events show up the most in participants' top recommendations
  * Explore trends amongst the set of users for which a given event is highly recommended
* NMF user-feature and item-feature matrices used to explore what factors may be most influencing participation and what makes a 'best event'
* Currently individual ALS models are performing better than the ensemble due to each predicting different subsets of the data better

**Business Observations**
* Opportunity to increase repeat participation of existing customers (brand loyalty), e.g. through targeted marketing using:
  * Events most frequently highly recommended
  * Set of people associated with a particular event included in their top recommendations
* More work and data are needed before explicit conclusions can be drawn from the clustering/NMF exercise (currently inconclusive)

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
