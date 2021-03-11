
def filter_visitor_days(threshold,
                        sensors_raw_data):
    """
    Shorts the full dataset (creates new one) depending on time window and error threshold
    @param threshold: error threshold for each day
    :param sensors_raw_data:
    """

    dayseps = []
    daylines = []
    date = ""
    for data_row in sensors_raw_data.itertuples():
        last_sensor = None
        current_sensor = data_row.SensorID
        for line in file:
            sep = separateLine(line)
            # first initialization
            if date == "":
                date = makeDate(sep["date"])
            # we have a new day
            if makeDate(sep["date"]) != date:
                if checkDayspan(dayseps, threshold):
                    for k in daylines:
                        short.write(k)
                dayseps = []
                daylines = []
            date = makeDate(sep["date"])
            # continue with new day or old day
            if fromDay <= date <= toDay:
                dayseps.append(sep)
                daylines.append(line)
        # check last day after last iteration
        if dayseps != [] and fromDay <= makeDate(dayseps[0]["date"]) <= toDay:
            if checkDayspan(dayseps, threshold):
                for k in daylines:
                    short.write(k)
        print("Dataset shorted and saved as \"shortdata\"")

    return filtered_sensor_data