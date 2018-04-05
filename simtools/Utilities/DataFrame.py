import numpy as np

from dtk.utils.observations.AgeBin import AgeBin

class NotUpsampleable(Exception): pass

def upsample_agebin(grouped_data, age_bin, aggregated_cols, weighted_cols, weighting_col):
    """
    Upsample a pandas DataFrame object containing a AgeBin column to the requested age_bin. Intended to be supplied
    as a dataframe groupby argument (to run on each group). It is ok for data outside the requested upsample range
    to be in this dataframe; it will simply be excluded in the result.
    Example usage: age_stratified_dataframe.groupby(['Year', 'Gender'].apply(upsample_agebin, AgeBin(15, 49))
    :param grouped_data: a pandas DataFrameGroupBy object, see above.
    :param age_bin: an AgeBin object representing inclusive lower and exclusive upper bounds.
    :param weighted_cols: columns in the grouped data/dataframe to do weighted sums of
    :return: A pandas DataFrame object with one row conataining the requested AgeBin-upsampled result
    """
    # Further notes:
    # verify we can do the requested upsample; this requires EXACT stitching of 'AgeBin' values to contain age_bin,
    # though data outside the requested range will be ignored here and in the upsaampling.
    if not AgeBin.can_upsample_bins(grouped_data['AgeBin'], age_bin):
        raise NotUpsampleable('Cannot upsample to age bin: %s . Data is missing.' % age_bin)

    # filter out data rows that are out of our requested age range, e.g. [50:55) is not in range of [15:49)
    filtered_df = grouped_data.loc[[age_bin.contains(ab) for ab in grouped_data['AgeBin']]]

    # grab row 0 and keep it as our base result; apply upsampled AgeBin
    result = grouped_data[0:1].reset_index(drop=True)
    result['AgeBin'] = str(age_bin)

    # aggregated data items
    total_weight = None
    for channel in aggregated_cols:
        total = np.sum(filtered_df[channel])
        result[channel] = total
        if channel == weighting_col: # hacky special case for use next
            total_weight = total

    # weighted sum items: model and reference data
    fraction = filtered_df[weighting_col] / total_weight
    for channel in weighted_cols:
        result[channel] = np.sum(fraction * filtered_df[channel])

    return result

