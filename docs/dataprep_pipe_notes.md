###Process Steps for Initial Data Mining / Pre-Prep:

* Save Access DB provided by Mergeo to data folder
* Open Access DB in ACCDB app
* Export Access DB as a postgres SQL file using the "Export SQL" option
  * Select filename and pick location of data folder
  * Leave all settings at default except add schema name of "public"
* As of 11/24 DB version:
  * Comment out the very last constraint (data not used and it throws an error in postgresql):
  * "/*ALTER TABLE "public"."SS_Teams" ADD CONSTRAINT "SS_TeamPeopleSS_Teams" FOREIGN KEY ("TeamNumber", "EventID") REFERENCES "public"."SS_TeamPeople" ("TeamID", "EventID") ON UPDATE SET NULL ON DELETE SET NULL;*/"
* In command line:
  * Connect to psql and create database: "CREATE DATABASE insertname;""
  * Quit psql
  * Load data from .sql file into newly created database, e.g. "psql mergeodb112417 < MergeoDB112417.sql &> 112417log.txt"
  * If any errors with this type of detail show up (Key (PersonID)=(10384) is not present in table "Persons"), you'll need to dig through the .sql file to identify the incorrectly numbered PersonID in the PersonEvents table and correct it.  If you cannot figure out the correct ID, comment out that record and let the Mergeo DB owner know either way so he can correct it.
* Update the main block in dataprep.py to call out the new postgresql database you want to use as your source data, and update the cleaned_df output with an appropriate name for this version.
* You are now ready to run the dataprep.py script to prepare and clean the dataset!
