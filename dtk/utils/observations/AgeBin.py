import re


class AgeBin:

    class InvalidAgeBinFormat(Exception): pass
    class NotMergeable(Exception): pass

    STR_FORMAT = '[%s%s%s)'
    # e.g. [15, 49) -> [(15)(, )(49))  delimiter must contain no numeric characters or '.'
    SPLIT_REGEX = re.compile('^\[(?P<start>[0-9.]+)(?P<delimiter>[^0-9.]+)(?P<end>[0-9.]+)\)$')
    DEFAULT_DELIMITER = ':'

    def __init__(self, start, end, delimiter=None):
        try:
            self.start = int(start)
        except ValueError:
            self.start = float(start)

        try:
            self.end = int(end)
        except ValueError:
            self.end = float(end)

        self.delimiter = delimiter or self.DEFAULT_DELIMITER

    def merge(self, other_bin):
        """
        Create a single AgeBin representing two adjacent AgeBins. Keeps delimiter of 'self'.
        :param other_bin: merge self with this other AgeBin object (self is lower age than other_bin)
        :return: an AgeBin object with delimiter set to self.delimiter (not other_bin.delimiter)
        """
        other_bin = other_bin if isinstance(other_bin, AgeBin) else AgeBin.from_string(other_bin)
        if self.end != other_bin.start:
            raise self.NotMergeable('AgeBin objects must be age-adjacent to be merged: %s %s' % (self, other_bin))
        return type(self)(start=self.start, end=other_bin.end, delimiter=self.delimiter)

    def contains(self, other_bin):
        """
        Is other_bin contained within the bounds of self?
        :param other_bin: an AgeBin object
        :return: True/False
        """
        other_bin = other_bin if isinstance(other_bin, AgeBin) else AgeBin.from_string(other_bin)
        return self.start <= other_bin.start and self.end >= other_bin.end

    def __str__(self):
        return self.STR_FORMAT % (self.start, self.delimiter, self.end)

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __ne__(self, other):
        return not (self == other)

    @classmethod
    def _split_string(cls, str):
        match = cls.SPLIT_REGEX.match(str)
        return match['start'], match['delimiter'], match['end']

    @classmethod
    def from_string(cls, str):
        try:
            start, delimiter, end = cls._split_string(str=str)
        except (KeyError, IndexError, TypeError) as e:
            example = cls(15,49)
            raise cls.InvalidAgeBinFormat('Required AgeBin format is e.g.: %s' % example)
        return cls(start=start, end=end, delimiter=delimiter)

    @classmethod
    def merge_bins(cls, bins):
        if len(bins) == 0:
            raise cls.NotMergeable('No AgeBins provided for merging.')

        # tolerant of string and object representations
        bins = [bin if isinstance(bin, AgeBin) else cls.from_string(bin) for bin in bins]
        bins = sorted(bins, key=lambda b: b.start)

        merged_bin = bins[0]
        for bin in bins[1:]:
            merged_bin = merged_bin.merge(bin)
        return merged_bin

    @classmethod
    def can_upsample_bins(cls, bins, target_bin):
        # tolerant of string and object representations
        bins = [bin if isinstance(bin, AgeBin) else cls.from_string(bin) for bin in bins]
        target_bin = target_bin if isinstance(target_bin, AgeBin) else cls.from_string(target_bin)

        # remove bins not within our target age range
        bins = [bin for bin in bins if target_bin.contains(bin)]

        # merge what is left over to see if it matches target_bin
        try:
            merged_bin = cls.merge_bins(bins)
        except cls.NotMergeable as e:
            return False

        return True if merged_bin == target_bin else False
