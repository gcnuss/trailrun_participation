# Trail Run Participation
Capstone project for Galvanize Data Science Immersive

### Business Understanding
There is a desire to understand customers’ patterns and factors that drive their participation in events.  The goal of this project is to understand customer participation habits in order to better inform marketing and race definition (location/timing/type) decisions, and to provide better support to customers in line with their interests if possible.  This will include exploration of geographic location (of event relative to home location of individual), when people register (day/time), customer tastes across event types, customer demographics, race characteristics, and associated patterns of participation (time of year, weekday/weekend, etc).

### Data Understanding
NW Trail Runs has an existing Access Database and database manager that I will work with to access the data and become familiar with it.  Currently I’ve taken an initial look at the database and there are sufficient attributes that I can work with.  There is data from several events that is not in the database yet but I’ve been in communication with the database manager and he will be adding the missing recent events in the next week.

### Data Preparation
I took a look at options to connect directly to MS Access so that I can get updated data whenever it is added, but the paths to do so are complex enough to not provide sufficient ROI for the effort on this project.  Instead, I have found an app for my Mac that will convert an Access database to a .sql file that I can import into postgres.  That is where I’ll start.  I’ll also need to do a fair bit of data cleaning and feature engineering to transform the existing registration and results data into the features needed to assess the factors of interest.

### Modeling
This project will include a fair bit of EDA and feature engineering upfront.  For modeling, I am considering either:
* a classification model for event participation (though it would have to be a single class model as I don’t have any records for the negative class)
* a regression model for race volume (though I have much less data from a race perspective, on the order of 100 or so, than participant data, which is up around 17,000)
* or a clustering and recommender system approach to looking at predicting participation (i.e. recommending future races to participants based on past participation).  This third option is currently my top choice to proceed with.
I could also apply hypothesis testing to further explore the level of influence of particular factors.

### Evaluation
The company has sufficient data to start with a train/validate/test split in order to quarantine test data for evaluating model performance.  I can use a variety of methods to further evaluate (log loss, reviewing confusion matrix / ROC curves with business understanding, etc if I do classification; will need to get).

### Deployment
Can you specify a minimal viable product (MVP) for presentation and deployment?
Well-polished Github readme; report to company on my methodology, assumptions, model(s), results, and recommendations for business decisions based on those results; provision of any models themselves for the company to continue to use in the future.

### Initial Database Review Notes (These are Rough….)
##### From Switchboard View:
**Events Table:**
Event Date, Event Name, Distance, Bib, FirstName, LastName, Sort (what does this represent?), Time, Start, Finish, Gender
I can easily get info to add Event Location, Event Type
Many events listed in the Access switchboard do not have results data imported yet

**Runners Table:**
Date, EventID, Event, Distance, Time, Bib, Age, Street Address, City, State, Zip, Phone, Email, Contact, Country, Notes, Gender, PersonID
Includes “DNS” for when a person registered but did not compete

**Series:**
Place, Best Three Score, Name, Age, Scores from Ind’l Race Dates, Series Category

##### From DB Relationships View:

**Persons Table** - PersonID (key), FirstName, LastName, Gender, Birthdate

**PersonEvents Table** - Person ID (key), EventID (key), Bib, Distance, Time, Start, Finish, Age, Registration Time, Total fee, Payment method, Phone, Street address, City, State, Emergency Contact Name / Phone, Zip, Country, Email, Contact, Tshirt, Tshirt Size, Hoodie, Hoodie Size, Notes

**Events Table** - Event ID (key), Event Name, Event Date, Notes, Mergeo Event

Also tables for EventDistances, DistanceSorts, SeriesEvents, Series, SeriesRules

**Current Breakdown of Loaded Events:**
Trail runs: 72/77 imported (working to get rest)
Street scrambles (urban nav races): 7/18 imported (working to get rest)
Nav races: 0/6 imported (tbd on adding)
Relays: 0/1 imported (tbd on adding)
Others: 2/7 imported (tbd on adding)
Currently have 16,181 participant entries total across imported events; adding the remaining trail runs and street scrambles will bring us close to 17,000 I believe.
