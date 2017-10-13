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
