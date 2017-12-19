#import from python modules:
import cPickle as pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import heapq

def create_attendance_report(cleaned_df_pkl):
    '''top level function that calls all other functions to create report contents
    and return results in an output file; takes a pickled pandas dataframe containing
    the full cleaned dataset, having been cleaned by dataprep.py'''

    with open (cleaned_df_pkl, 'rb') as f:
        df = pickle.load(f)

    repeat_attendees_df = create_repeat_attendees_df(df)

    print_attendee_counts(repeat_attendees_df)

    events_df = create_events_df

def create_repeat_attendees_df(df):
    '''Creates a new dataframe with a row for each unique PersonID and information
    on all events they've attended, their attendance trends in terms of repeat
    attendances, etc.  Takes the baseline dataframe loaded in the top level function
    and returns the "repeat_attendees_df"'''

    #create initial dataframe with person ID and # of events they have attended
    repeat_attendees_df = pd.DataFrame.from_dict({
                'PersonID': df.groupby(by='PersonID').count().index.values,
                'Event_Count': df.groupby(by='PersonID').count()['EventID'].values})

    #mine event IDs, dates, and associated series IDs from baseline df and add to new df
    repeat_attendees_df = mine_repeat_attendees_details(df, repeat_attendees_df)

    #format dates to be truncated to day
    repeat_attendees_df['Event_Dates'] = repeat_attendees_df['Event_Dates'].apply(lambda x: date_trunc(x))

    #calculate lag time between first and second event attended (< or > 6 mo)
    repeat_attendees_df['Repeat_Lag_Time'] = repeat_attendees_df['PersonID'].apply(lambda x: repeat_lag_time(x, repeat_attendees_df))

    #identify whether repeat attendees attended more than one series or not
    repeat_attendees_df['Cross_Series_Attend'] = repeat_attendees_df['Series_IDs'].apply(
                                lambda x: 'One Off'
                                if len(x) == 1
                                else ('Single - {}'.format(x[0]) if len(set(x)) == 1 else 'Multiple'))

    return repeat_attendees_df

def mine_repeat_attendees_details(df, repeat_attendees_df):
    '''Gathers data on eventIDs, event dates, and seriesIDs for each person's
    attendance records'''

    personID_eventID_dict = defaultdict(list)
    personID_eventdate_dict = defaultdict(list)
    personID_seriesID_dict = defaultdict(list)

    for num in df.groupby(by='PersonID').count().index.values:
        personID_eventID_dict[num].append(list(df[df['PersonID']==num]['EventID'].values))
        personID_eventdate_dict[num].append(list(df[df['PersonID']==num]['Event_Date'].values))
        personID_seriesID_dict[num].append(list(df[df['PersonID']==num]['SeriesID'].values))

    repeat_attendees_df['Event_IDs'] = repeat_attendees_df['PersonID'].map(personID_eventID_dict).apply(lambda x: x[0])
    repeat_attendees_df['Event_Dates'] = repeat_attendees_df['PersonID'].map(personID_eventdate_dict).apply(lambda x: x[0])
    repeat_attendees_df['Series_IDs'] = repeat_attendees_df['PersonID'].map(personID_seriesID_dict).apply(lambda x: x[0])

    return repeat_attendees_df

def date_trunc(date_list):
    '''takes a date in numpy datetime64 format and truncates it down to the day
    level only'''
    return [np.datetime64(item, 'D') for item in date_list]

def repeat_lag_time(person_ID, repeat_attendees_df, duration_days=183):
    '''Identifies for each person whether they are one off attendees, or if they
    are repeat attendees whether the lag time between their first and second event
    is less or more than 6 months'''

    if person_ID in repeat_attendees_df['PersonID'].values:
        date_list = sorted(repeat_attendees_df[repeat_attendees_df['PersonID'] == person_ID]['Event_Dates'].values[0])
        if len(date_list) == 1:
            return 'One Off'
        elif len(date_list) > 1:
            first_event, second_event = heapq.nsmallest(2, date_list)
            delta = second_event - first_event
            if delta < np.timedelta64(duration_days, 'D'):
                return 'Under {} days'.format(duration_days)
            else:
                return 'Over {} days'.format(duration_days)
    else:
        print 'Error in repat_lag_time: Provided PersonID does not exist in the dataframe'

def print_attendee_counts(repeat_attendees_df):
    '''calculates various sums and provides data on counts of repeat attendees
    and how many attend a single series vs. multiple series'''

    total_repeats_count = float(len(repeat_attendees_df[repeat_attendees_df['Event_Count']>1]))

    single_series = float(len(repeat_attendees_df[repeat_attendees_df['Cross_Series_Attend'] > 'Single']))

    multi_series = len(repeat_attendees_df[repeat_attendees_df['Cross_Series_Attend'] == 'Multiple'])

    single_attend_count = len(repeat_attendees_df[repeat_attendees_df['Cross_Series_Attend'] == 'One Off'])

    total_people = float(len(repeat_attendees_df))

    print ("{} percent of attendees are repeat. {} percent of repeat attendees \
    have only attended one series and {} percent have attended multiple".format(
                            round((single_series+multi_series)/total_people*100),
                            round(single_series/total_repeats_count * 100),
                            round(multi_series/total_repeats_count * 100)))

    multi_attend_single_series_counts = repeat_attendees_df[repeat_attendees_df['Cross_Series_Attend'] > 'Single'].groupby(by='Cross_Series_Attend')['Event_Count'].count()

    print ('For the {} Repeat Attendees all in One Series, Breakdown is:'.format(int(single_series)))

    for idx, val in enumerate(multi_attend_single_series_counts):
        print ('Series {}: {}%'.format(idx, round(val/single_series * 100)))

def create_events_df(df, repeat_attendees_df):
    '''TBD'''

    events_df = pd.DataFrame.from_dict(
        {'EventID': df.groupby(by='EventID').count().index.values,
        'Total_Part_Count': df.groupby(by='EventID').count()['PersonID'].values})

    events_df = mine_event_details(df, events_df)

    events_df = add_event_data_counts_to_df(repeat_attendees_df, events_df)

    events_df.sort_values(by='Event_Date', ascending=True, inplace=True)

    events_df['%_Repeat_Under_Six'] = 100 * events_df['Repeat_Under_Six_Mo'] / (events_df['First_Timers'])
    events_df['%_Repeat_Under_Six'] = events_df['%_Repeat_Under_Six'].apply(lambda x: int(x))

    events_df['%_Repeat_Over_Six'] = 100 * events_df['Repeat_Over_Six_Mo'] / (events_df['First_Timers'])
    events_df['%_Repeat_Over_Six'] = events_df['%_Repeat_Over_Six'].apply(lambda x: int(x))

    events_df.reset_index(drop=True, inplace=True)

    return events_df

def mine_event_details(df, events_df):
    '''TBD'''

    eventID_personID_dict = defaultdict(list)
    eventID_eventdate_dict = defaultdict(list)
    eventID_eventname_dict = defaultdict(list)
    eventID_seriesID_dict = defaultdict(list)
    eventdate_dict = defaultdict(list)
    eventname_dict = defaultdict(list)
    seriesID_dict = defaultdict(list)

    for num in event_person_count_df.index.values:
        eventID_personID_dict[num].append(list(df[df['EventID']==num]['PersonID'].values))
        eventID_eventdate_dict[num].append(list(df[df['EventID']==num]['Event_Date'].values))
        eventID_eventname_dict[num].append(list(df[df['EventID']==num]['Event_Name'].values))
        eventID_seriesID_dict[num].append(list(df[df['EventID']==num]['SeriesID'].values))

        eventdate_dict[num] = eventID_eventdate_dict[num][0][0]
        eventname_dict[num] = eventID_eventname_dict[num][0][0]
        seriesID_dict[num] = eventID_seriesID_dict[num][0][0]

    events_df['Person_IDs'] = events_df['EventID'].map(eventID_personID_dict).apply(lambda x: x[0])
    events_df['Event_Date'] = events_df['EventID'].map(eventdate_dict).apply(lambda x: np.datetime64(x, 'D'))
    events_df['Event_Name'] = events_df['EventID'].map(eventname_dict)
    events_df['Series_ID'] = events_df['EventID'].map(seriesID_dict)

    return events_df

def add_event_data_counts_to_df(repeat_attendees_df, events_df):
    '''TBD'''

    event_data_dict = defaultdict(list)

    for event in events_df['EventID'].values:

        repeats_multiseries, repeats_singleseries, single_attend = get_single_event_data_counts(event, repeat_attendees_df, events_df)

        event_data_dict['repeat_singleseries'].append(repeats_singleseries)
        event_data_dict['repeat_multiseries'].append(repeats_multiseries)
        event_data_dict['single_attend'].append(single_attend)

        first_timer, first_timer_ids, under_six_month_repeat, over_six_month_repeat = get_single_event_first_timer_counts(event, repeat_attendees_df, events_df)

        event_data_dict['first_timer'].append(first_timer)
        event_data_dict['six_mo_repeat'].append(under_six_month_repeat)
        event_data_dict['longer_repeat'].append(over_six_month_repeat)
        event_data_dict['first_timer_ids'].append(first_timer_ids)

    events_df['Repeat_SingleSeries'] = event_data_dict['repeat_singleseries']
    events_df['Repeat_MultiSeries'] = event_data_dict['repeat_multiseries']
    events_df['Single_Attend'] = event_data_dict['single_attend']
    events_df['First_Timers'] = event_data_dict['first_timer']
    events_df['First_Timer_IDs'] = event_data_dict['first_timer_ids']
    events_df['Repeat_Under_Six_Mo'] = event_data_dict['six_mo_repeat']
    events_df['Repeat_Over_Six_Mo'] = event_data_dict['longer_repeat']

    return events_df

def get_single_event_data_counts(event, repeat_attendees_df,
                                events_df, repeats_singleseries = 0,
                                repeats_multiseries = 0, single_attend = 0):

    for person in events_df[events_df['EventID'] == event]['Person_IDs'].values[0]:
        att_status = repeat_attendees_df[
            repeat_attendees_df['PersonID'] == person]['Cross_Series_Attend'].values[0]
        lag_status = repeat_attendees_df[
            repeat_attendees_df['PersonID'] == person]['Repeat_Lag_Time'].values[0]
        if att_status == 'Multiple':
            repeats_multiseries += 1
        elif att_status == 'One Off':
            single_attend += 1
        else:
            repeats_singleseries += 1

    return repeats_multiseries, repeats_singleseries, single_attend

def get_single_event_first_timer_counts(event, repeat_attendees_df, events_df,
                                        first_timer = 0, first_timer_ids = list(),
                                        under_six_month_repeat = 0,
                                        over_six_month_repeat = 0):

    if min(repeat_attendees_df[
        repeat_attendees_df['PersonID']==person][
        'Event_Dates'].values[0]) == events_df[
        events_df['EventID']==event]['Event_Date'].values[0]:
        first_timer += 1
        first_timer_ids.append(person)
        if lag_status == 'Under 183 days':
            under_six_month_repeat += 1
        elif lag_status == 'Over 183 days':
            over_six_month_repeat += 1

    return first_timer, first_timer_ids, under_six_month_repeat, over_six_month_repeat

if __name__ == "__main__":
    create_attendance_report('../data/cleaned_df_24NOV17.pkl')
