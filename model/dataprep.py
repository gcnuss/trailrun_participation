import psycopg2
import pandas as pd
import numpy as np
from collections import defaultdict
import geocoder

class TrailDataPrep(object):
    '''This class initiates a psycopg2 session to connect to the postgres db
    containing all of the data, runs a query to create a single set of results
    containing all columns to be used from modeling across several tables in the
    db, creates an initial dataset in a pandas df, and has several options for
    cleaning the data to prepare for further EDA and modeling

    Inputs: database name and host for postgres db

    Parameters:
        postgres database name
        host for postgres session

    Attributes:
        query_results: the output from the combined psql query
        base_dataset: initial pandas df with raw data from the postgres db
        clean_dataset: cleaned pandas df that all cleaning actions are performed on
        event_nm_ven_dict: dictionary of event names and venues
        event_dets = list of tuples with each tuple containing event name, venue,
                        geocoder.google address instance, and zip codes
    '''


    def __init__(self, dbname='mergeoruns', host='localhost'):
        self.dbname = dbname
        self.host = host
        self.cur = None
        self.query_results = None
        self.base_dataset = None
        self.clean_dataset = None
        self.event_nm_ven_dict = None
        self.event_dets = None

    def init_psql_session(self):
        conn = psycopg2.connect(dbname=self.dbname, host=self.host)
        self.cur = conn.cursor()
        conn.autocommit = True

    def data_coll_query(self):
        '''pre-defined query based on my data review and determination of what
        columns to include; returns a list of records'''
        query = '''
            WITH master_temp
            AS (SELECT pe."PersonID", pe."EventID", pe."Distance", pe."Time", pe."Age",
            pe."Registration time", pe."Total fee", pe."Payment method",
            pe."Street Address", pe."City", pe."State/Province", pe."Emergency contact name",
            pe."Zip Code", pe."Country", pe."Contact", pe."Tshirt", pe."Hoodie",
            e."Event_Name", e."Event_Date", e."MergeoEvent", e."EventTypeID", se."SeriesID"
            FROM "PersonEvents" pe
            LEFT JOIN "Events" e ON pe."EventID" = e."EventID"
            LEFT JOIN "SeriesEvents" se ON pe."EventID" = se."EventID")
            SELECT mt."PersonID", mt."EventID", mt."Age", p."Gender", mt."Distance",
            ds."Miles", mt."Time", mt."Total fee", sstp."Prereg", mt."Registration time",
            mt."Payment method", mt."Street Address", mt."City", mt."State/Province",
            mt."Zip Code", mt."Country", mt."Emergency contact name", mt."Contact",
            mt."Tshirt", mt."Hoodie", mt."Event_Name", mt."Event_Date", mt."MergeoEvent",
            mt."EventTypeID", et."EventType", mt."SeriesID", s."Series", sstp."HowHeard"
            FROM master_temp mt
            LEFT JOIN "DistanceSorts" ds ON LOWER(mt."Distance") = LOWER(ds."Distance")
            LEFT JOIN "EventTypes" et ON mt."EventTypeID" = et."EventTypeID"
            LEFT JOIN "Persons" p ON p."PersonID" = mt."PersonID"
            LEFT JOIN "SS_TeamPeople" sstp ON mt."PersonID" = sstp."PersonID"
            AND mt."EventID" = sstp."EventID"
            LEFT JOIN "Series" s ON mt."SeriesID" = s."SeriesID"
            '''
        self.cur.execute(query)
        self.query_results = self.cur.fetchall()
        return self.query_results

    def create_df(self):
        '''creates raw data pandas dataframe with pre-defined columns aligning
        with the query columns included'''
        cols = ["PersonID", "EventID", "Age", "Gender", "Distance", "Miles", "Time",
        "Total fee", "SS_Prereg", "Registration time", "Payment method",
        "Street Address", "City", "State/Province", "Zip Code", "Country",
        "Emergency contact name", "Contact", "Tshirt", "Hoodie", "Event_Name", "Event_Date",
        "MergeoEvent", "EventTypeID", "EventType", "SeriesID", "Series",
        "HowHeard"]
        self.base_dataset = pd.DataFrame(self.query_results, columns = cols)
        return self.base_dataset

    def col_cleaning(self):
        '''completes initial cleaning on data columns and some basic feature
        engineering; fill na's for columns with missing values; drop old columns
        where new separate columns are created; separate functions called when
        needed for add'l pre-processing'''

        self.clean_dataset = self.base_dataset.copy()

        self.clean_dataset['PersonID'] = self.clean_dataset['PersonID'].apply(lambda x: int(x))

        #Age Column
        self.get_ages()
        self.clean_dataset['Age2'].fillna(value=int(
                                self.clean_dataset['Age2'].mean()), inplace=True)
        self.clean_dataset = self.clean_dataset.drop('Age', axis=1)
        #Distance Column
        self.clean_dataset['Distance'].fillna(value='Variable-SS', inplace=True)
        #Miles Column
        self.get_miles()
        self.clean_dataset['Miles2'].fillna(value=round(
                                    self.clean_dataset['Miles2'].mean(),0),
                                    inplace=True)
        self.clean_dataset = self.clean_dataset.drop('Miles', axis=1)
        #Gender Column
        self.clean_dataset['Gender'].fillna(value='Other', inplace=True)
        #Time Column
        self.clean_times()
        self.clean_dataset = self.clean_dataset.drop('Time', axis=1)
        #Total fee Column
        self.clean_dataset['Total fee'] = pd.to_numeric(
                                self.clean_dataset['Total fee'], errors='coerce')
        self.clean_dataset['Total fee'].fillna(value=round(
                                self.clean_dataset['Total fee'].mean(),2), inplace=True)
        #Preregistration Y/N, pulled from SS_Prereg, Registration time Columns
        self.clean_dataset['Preregistered'] = self.clean_dataset[['SS_Prereg',
                                'Registration time']].apply(lambda row: 1 if (
                                (row[0]) or (pd.notnull(row[1]))) else 0, axis=1)
        self.clean_dataset.drop(['SS_Prereg', 'Registration time'], axis=1, inplace=True)
        #Emergency Contact Y/N, pulled from Emergency contact name Column
        self.clean_dataset['has_emerg_contact'] = self.clean_dataset[[
                                'Emergency contact name']].apply(lambda row: 1
                                if pd.notnull(row[0])
                                else 0, axis=1)
        self.clean_dataset.drop('Emergency contact name', axis = 1, inplace=True)
        #Tshirt Column
        self.clean_dataset['Tshirt'].fillna(value='N', inplace=True)
        #Hoodie Column
        self.clean_dataset['Hoodie'].fillna(value='N', inplace=True)
        #SeriesID Column
        self.clean_series_ids()
        self.clean_dataset['HasSeries'] = self.clean_dataset[['SeriesID']].apply(
                                lambda row: 'Y' if pd.notnull(row[0])
                                else 'N', axis=1)
        self.clean_dataset['SeriesID'].fillna(value=0., inplace=True)
        #Series Column, drop as duplicate to SeriesID Column (perfectly co-linear)
        self.clean_dataset.drop('Series', axis=1, inplace=True)
        #HowHeard Column, drop, not enough data to use (99 out of almost 17000)
        self.clean_dataset.drop('HowHeard', axis=1, inplace=True)
        #Payment method Column
        self.clean_pay_methods()
        #Contacts Column
        self.clean_contacts()


    def get_ages(self):
        '''find ages for PersonIDs that exist in some records but are missing
        from others; for use in populating missing age values'''

        persons_no_age = self.base_dataset[pd.isnull(self.base_dataset['Age'])]['PersonID'].values
        persIDs_ages = zip(list(self.base_dataset['PersonID'].values), list(self.base_dataset['Age'].values))
        D_age = defaultdict(list)

        for persID, age in persIDs_ages:
            if pd.notnull(age):
                D_age[persID].append(int(age))

        D2_age = defaultdict(list)

        for person in persons_no_age:
            ages_list = D_age.get(person)
            if ages_list is None:
                D2_age[person] = np.NaN
            elif len(ages_list) == 0:
                D2_age[person] = np.NaN
            else:
                D2_age[person] = int(np.round(np.mean(ages_list), decimals=0))

        self.clean_dataset['Age2'] = self.clean_dataset[['PersonID', 'Age']].apply(
                                lambda row: int(row[1]) if pd.notnull(row[1])
                                else D2_age[row[0]], axis=1)

    def get_miles(self):
        '''identify mileages for Miles column for records that have valid distance
        values but are missing mileage values'''
        dist_dict = {'1/2 Marathon':13., 'Half Marathon early start':13.,
        '5k early start':3., '5 Mile late start': 5., '10k early start':6.,
        'Half Marathon late start':13., 'Variable-SS':np.NaN}

        self.clean_dataset['Miles2'] = self.clean_dataset[['Distance', 'Miles']].apply(
                                    lambda row: row[1] if pd.notnull(row[1]) else dist_dict[row[0]], axis=1)

    def clean_times(self):
        '''Change times for all street scrambles to 90 minutes; note that some
        of these are 3 hr or 2 hr but I don't have the granularity to see that
        in the data right now; follow up item for Dan; also drop records that
        are missing times and are not street scrambles'''

        self.clean_dataset['Time2'] = self.clean_dataset[['EventID', 'Time']].apply(
                                    lambda row: '1:30:00.0' if row[0] > 999
                                    else row[1], axis=1)
        records_to_drop = list(self.clean_dataset[pd.isnull(
                                    self.clean_dataset['Time2'])].index)

        self.clean_dataset.drop(labels=records_to_drop, inplace=True)

    def clean_pay_methods(self):
        '''payment methods contains a variety of oddball entries; this function
        buckets down to a handful of useful categories (cash, credit, check,
        paypal, comp, and other); also creates a "has_pay_method" feature to
        account for all of the None values'''

        self.clean_dataset['Payment method'] = self.clean_dataset[[
                                    'Payment method']].apply(lambda row: 'cash'
                                    if (row[0] == 'Cash' or row[0] == 'cash/comp')
                                    else row[0], axis=1)
        self.clean_dataset['Payment method'] = self.clean_dataset[[
                                    'Payment method']].apply(lambda row: 'check'
                                    if (row[0] == 'Check')
                                    else row[0], axis=1)
        self.clean_dataset['Payment method'] = self.clean_dataset[[
                                    'Payment method']].apply(lambda row: 'credit'
                                    if (row[0] == 'Authorize' or row[0] == 'CC'
                                    or row[0] == 'cc' or row[0] == '$261 CC, $30 cash')
                                    else row[0], axis=1)
        self.clean_dataset['Payment method'] = self.clean_dataset[[
                                    'Payment method']].apply(lambda row: 'paypal'
                                    if (row[0] == 'PayPal' or row[0] == 'PayPal pending')
                                    else row[0], axis=1)
        self.clean_dataset['Payment method'] = self.clean_dataset[[
                                    'Payment method']].apply(lambda row: 'comp'
                                    if (row[0] == 'Comp')
                                    else row[0], axis=1)
        self.clean_dataset['Payment method'] = self.clean_dataset[[
                                    'Payment method']].apply(lambda row: 'other'
                                    if (row[0] != 'cash' and row[0] != 'check'
                                    and row[0] != 'comp' and row[0] != 'credit'
                                    and row[0] != 'paypal' and row[0] != None)
                                    else row[0], axis=1)
        self.clean_dataset['has_pay_method'] = self.clean_dataset[[
                                    'Payment method']].apply(lambda row: 'N'
                                    if row[0] == None
                                    else 'Y', axis=1)

    def clean_contacts(self):
        '''similar to payment method, the contacts column contains a variety of
        oddball entries; bucketing here to Yes, No, and No Response.  Need to
        follow up with Dan on what the -1 and 0 values mean; for no bucketed in
        No Response'''

        self.clean_dataset['Contact'] = self.clean_dataset[['Contact']].apply(
                                        lambda row: 'No'
                                        if (row[0] == 'no' or row[0] == 'NO')
                                        else row[0], axis=1)
        self.clean_dataset['Contact'] = self.clean_dataset[['Contact']].apply(
                                        lambda row: 'No Response'
                                        if (row[0] == '-1' or row[0] == '0'
                                        or row[0] == None)
                                        else row[0], axis=1)

    def clean_series_ids(self):
        '''SeriesID column has separate series numbers for the same series in
        different years.  This function makes the series IDs year-independent
        such that it is only looking at the type of series over multiple years'''

        #align winter series from 2015/2016 and 2016/2017 under ID 1
        self.clean_dataset['SeriesID'] = self.clean_dataset[['SeriesID']].apply(
                                lambda row: 1. if row[0] == 4. else row[0], axis=1)
        #align half marathon series from 206 and 2017 under ID 2
        self.clean_dataset['SeriesID'] = self.clean_dataset[['SeriesID']].apply(
                                lambda row: 2. if row[0] == 5. else row[0], axis=1)
        #align trail to grill series from 2016 and 2017 under ID 3
        self.clean_dataset['SeriesID'] = self.clean_dataset[['SeriesID']].apply(
                                lambda row: 3. if row[0] == 6. else row[0], axis=1)
        #assign a series ID to Street Scrambles, set under ID 4
        self.clean_dataset['SeriesID'] = self.clean_dataset[[
                                'EventType', 'SeriesID']].apply(
                                lambda row: 4. if row[0] == 'Street Scramble'
                                else row[1], axis=1)

    def engr_features(self):
        '''Adds some features to the data and transforms some as well.  Calls
        separate functions where appropriate for more involved feature engineering.
        Running this function runs all feature engineering functions.'''

        self.add_venue_zip()



    def add_venue_zip(self):
        '''Gets venue addresses from geocoder, extracts zip codes, and adds them
        to the clean_dataset; future improvements: have db owner add the venue
        information to the original dataset; add exceptions to this function to
        better handle errors.'''

        event_names = list(self.clean_dataset['Event_Name'].unique())
        event_venues = ["Fort Ebey State Park, Coupeville, WA",
        "Interlaken Park, Seattle, WA", "Ravenna Park, Seattle, WA",
        "Carkeek Park, Seattle, WA", "Rattlesnake Ridge, North Bend, WA",
        "Cedar Mountain, Renton, WA", "Tiger Mountain, Issaquah, WA",
        "Redmond Watershed, Redmond, WA", "Interlaken Park, Seattle, WA",
        "Ravenna Park, Seattle, WA", "Carkeek Park, Seattle, WA",
        "Seward Park, Seattle, WA", "Soaring Eagle Regional Park, Sammamish, WA",
        "Redmond Watershed, Redmond, WA", "Soaring Eagle Regional Park, Sammamish, WA",
        "St. Edward State Park, Kenmore, WA", "Redmond Watershed, Redmond, WA",
        "Carkeek Park, Seattle, WA", "Lake Sammamish State Park, Issaquah, WA",
        "Seward Park, Seattle, WA", "Cle Elum, WA", "St. Edward State Park, Kenmore, WA",
        "St. Edward State Park, Kenmore, WA", "Market Theater, Seattle, WA",
        "Lord Hill Regional Park, Snohomish, WA", "Eddon Boat Park, Gig Harbor, WA",
        "Landsburg park, maple valley, wa", "Redmond Watershed, Redmond, WA",
        "Sprague, WA", "Paradise Valley Conservation Area, Woodinville, WA",
        "Wilburton Hill Park, Bellevue, WA", "Woodland Park, Seattle, WA",
        "Seward Park, Seattle, WA", "Cougar Mountain Trailhead, Bellevue, WA",
        "Alki Beach, West Seattle, WA", "Juanita Beach Park, Kirkland, WA",
        "Market Theater, Seattle, WA", "Redmond Town Center, Redmond, WA",
        "Fremont Sunday Flea Market, Fremont, Seattle, WA",
        "Eddon Boat Park, Gig Harbor, WA", "Lake Sammamish State Park, Issaquah, WA",
        "Cle Elum-Roslyn High School, Cle Elum, WA", "Interlaken Park, Seattle, WA",
        "Everett Mall, Everett, WA", "Fremont Sunday Flea Market, Fremont, Seattle, WA",
        "Nordic Heritage Museum, Ballard, Seattle, WA"]

        if len(event_names) != len(event_venues):
            print('Error in add_venue_zip; event_venues length does not match\
                    event_names length; check data before proceeding.')
            break
        else:
            continue

        self.event_nm_ven_dict = dict(zip(event_names, event_venues))

        self.event_dets = []
        for name, venue in event_nm_ven:
            g = geocoder.google(venue)
            zipcode = g.postal
            self.event_dets.append((name, venue, g, zipcode))
            time.sleep(0.1)

        D_zips = defaultdict()
        for name, venue, g, zipcode in event_dets:
            D_zips[name] = int(zipcode)

        clean_dataset['Venue_Zip'] = clean_dataset['Event_Name'].apply(lambda x: D_zips[x])

if __name__ == "__main__":
    dataprep = TrailDataPrep(dbname='mergeoruns101717', host='localhost')
    dataprep.init_psql_session()
    dataprep.data_coll_query()
    dataprep.create_df()
    dataprep.col_cleaning()
    cleaned_df = dataprep.clean_dataset
    with open ('cleaned_df.pkl', 'w') as f:
        pickle.dump(cleaned_df, f)
