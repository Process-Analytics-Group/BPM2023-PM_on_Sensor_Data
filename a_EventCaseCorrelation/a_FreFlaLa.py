"""
Code from Master Project
"""
import a_EventCaseCorrelation.FreFraLa.Filter as FreFraLa_Filter


def apply_threshold_filtering():
    # filter input dataset for errors and time
    FreFraLa_Filter.filter_visitor_days()

