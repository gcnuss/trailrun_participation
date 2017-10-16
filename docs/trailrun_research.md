What are our customers' tastes regarding types of events?
- Clustering or other unsupervised learning if data available to ID key factors
- Clustering of attendance by event type; way to look at overlaps??

What are their patterns of participation in our events?
- Classification to predict participation; then look at features that contribute the most
- Not really ML, but could definitely do EDA to identify participation patterns by demographics (M/F, age ranges, past participation, time of year, location, etc)
  - Could then build a predictive model for participation based on key factors

Where do they live, and how does geography affect event participation?
- Hypothesis testing for whether geography affects participation
- Assuming it does, could look at geography predictor (e.g. distance from home to event) for predicting participation

What other events do they participate in?
- Web scraping of participants by name (?) from results of other companies' races
- EDA to add features that identify other Mergeo event types attended and using web scraping also what other event types are participated in
- SQL DB mining to optimize the data we have from Dan and look at joins, etc (?)

What times of day and days of the week do people tend to register for events?
- Classification model to predict registration day (or linear regression to predict day & time?)
- Definite opportunity for EDA to look at what we have in the existing data for peaks / valleys (then will need to do ML to understand what predictors drive those peaks / valleys)

What email and social media messages do people respond to and engage with the most strongly?
- A/B hypothesis testing
- Multi armed bandit bayesian statistics; bayesian ML techniques / algorithms (different options may work better for different demographics)

Could also look at churn prediction, define a person as inactive if no events after X amount of time, and then look at what drives people to remain active vs. inactive!

Questions:
- Format of database? Access Database
https://wiki.postgresql.org/wiki/Microsoft_Access_to_PostgreSQL_Conversion
http://www.postgresonline.com/journal/archives/24-Using-MS-Access-with-PostgreSQL.html
- What types of data are available? Registration and Results Data (not sure if separate or together; registered but no shows not sure if tracked);
- Security of data, e.g. running things on AWS, etc?  I'll just need to make sure I keep individuals' data protected
- How much can I share for my presentation (e.g. leaving name of company out, etc)?  Listing NW Trail Runs is ok, no worries there
- Scope for project vs. stretch goals?  Discussed priorities; Eric will provide more feedback on this after talking to Gretchen too; focus first on what we can learn about customers' habits from existing data (geographic, patterns of participation, timing of registration, customer tastes across all Mergeo/NW Trail Run events)
