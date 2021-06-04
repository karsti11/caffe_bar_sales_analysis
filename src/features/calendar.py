import pandas as pd

easter_dates = ['2013-03-31', '2014-04-20', '2015-04-05', '2016-03-27', 
                '2017-04-16', '2018-04-01', '2019-04-21', '2020-04-12']

easter_monday_dates = pd.to_datetime(easter_dates) + pd.to_timedelta(1, unit='d')