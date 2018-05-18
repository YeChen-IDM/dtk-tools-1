import unittest

from dtk.utils.observations.AgeBin import AgeBin

class TestAgeBin(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_instantiation(self):
        ab = AgeBin(15,49)
        self.assertEqual(ab.start, 15)
        self.assertEqual(ab.end, 49)
        self.assertEqual(ab.delimiter, AgeBin.DEFAULT_DELIMITER)

        ab = AgeBin(0,99,delimiter=', ')
        self.assertEqual(ab.start, 0)
        self.assertEqual(ab.end, 99)
        self.assertEqual(ab.delimiter, ', ')

    # merge tests

    def test_merge_works_with_AgeBin_and_string(self):
        expected_ab = AgeBin(10, 99)

        ab1 = AgeBin(10, 15)
        ab2 = "[15:99)"
        self.assertEqual(ab1.merge(ab2), expected_ab)

        ab2 = AgeBin.from_string(ab2)
        self.assertEqual(ab1.merge(ab2), expected_ab)

    def test_merge_raises_if_not_consecutive_ages(self):
        ab1 = AgeBin(10, 15)
        ab2 = AgeBin(100, 200)
        self.assertRaises(AgeBin.NotMergeable, ab1.merge, other_bin=ab2)
        self.assertRaises(AgeBin.NotMergeable, ab2.merge, other_bin=ab1)

        ab2=AgeBin(5, 10)
        # wrong order
        self.assertRaises(AgeBin.NotMergeable, ab1.merge, other_bin=ab2)
        # right order
        expected_ab = AgeBin(5, 15)
        self.assertEqual(ab2.merge(ab1), expected_ab)

    def test_merge_sets_proper_delimiter(self):
        ab1 = AgeBin(5, 10, delimiter='###')
        ab2 = AgeBin(10, 15)
        merged = ab1.merge(ab2)
        self.assertNotEqual(ab1.delimiter, ab2.delimiter)
        self.assertEqual(merged.delimiter, '###')

    # contains tests

    def test_contains_works_with_AgeBin_and_string(self):
        big_ab = AgeBin(10, 99)

        ab = "[10, 15)"
        self.assertTrue(big_ab.contains(ab))
        ab = AgeBin.from_string(ab)
        self.assertTrue(big_ab.contains(ab))

    def test_contains_works_properly(self):
        big_ab = AgeBin(10, 99)

        # testing a variety of edge cases, both 'contained' and not 'contained'
        ab = AgeBin(0, 9)
        self.assertFalse(big_ab.contains(ab))
        ab = AgeBin(0, 10)
        self.assertFalse(big_ab.contains(ab))
        ab = AgeBin(0, 11)
        self.assertFalse(big_ab.contains(ab))
        ab = AgeBin(98, 200)
        self.assertFalse(big_ab.contains(ab))
        ab = AgeBin(99, 200)
        self.assertFalse(big_ab.contains(ab))
        ab = AgeBin(100, 200)
        self.assertFalse(big_ab.contains(ab))

        ab = AgeBin(10, 99)
        self.assertTrue(big_ab.contains(ab))
        self.assertTrue(ab.contains(big_ab))
        ab = AgeBin(10, 15)
        self.assertTrue(big_ab.contains(ab))
        ab = AgeBin(15, 30)
        self.assertTrue(big_ab.contains(ab))
        ab = AgeBin(90, 99)
        self.assertTrue(big_ab.contains(ab))
        self.assertFalse(ab.contains(big_ab)) # and check the inverse case...

    # equality/inequality method tests

    def test_equality_comparison(self):
        ab1 = AgeBin(0, 99)
        ab2 = AgeBin(0, 99)
        self.assertTrue(ab1 == ab2)
        ab2 = AgeBin(0, 98)
        self.assertFalse(ab1 == ab2)
        ab2 = AgeBin(1, 99)
        self.assertFalse(ab1 == ab2)

    def test_inequality_comparison(self):
        ab1 = AgeBin(0, 99)
        ab2 = AgeBin(0, 99)
        self.assertFalse(ab1 != ab2)
        ab2 = AgeBin(0, 98)
        self.assertTrue(ab1 != ab2)
        ab2 = AgeBin(1, 99)
        self.assertTrue(ab1 != ab2)

    # from_string tests

    def test_from_string_raises_if_format_is_invalid(self):
        self.assertRaises(AgeBin.InvalidAgeBinFormat, AgeBin.from_string, '[009)')
        self.assertRaises(AgeBin.InvalidAgeBinFormat, AgeBin.from_string, '[0:9d)')
        self.assertRaises(AgeBin.InvalidAgeBinFormat, AgeBin.from_string, '[d0:9)')
        self.assertRaises(AgeBin.InvalidAgeBinFormat, AgeBin.from_string, '[0:9]')
        self.assertRaises(AgeBin.InvalidAgeBinFormat, AgeBin.from_string, '(0:9)')

    def test_from_string_works_properly(self):
        ab_string = '[0:;:99)'
        ab = AgeBin.from_string(ab_string)
        self.assertEqual(ab.start, 0)
        self.assertEqual(ab.end, 99)
        self.assertEqual(ab.delimiter, ':;:')

    # merge_bins tests

    def test_merge_bins_works_with_AgeBin_and_string(self):
        bins = ['[0:49)', AgeBin(49, 99)]
        merged = AgeBin.merge_bins(bins=bins)
        expected = AgeBin(0,99)
        self.assertEqual(merged, expected)

    def test_merge_bins_works_with_unsorted_AgeBins(self):
        bins = [AgeBin(49, 99), AgeBin(0,49)]
        merged = AgeBin.merge_bins(bins=bins)
        expected = AgeBin(0,99)
        self.assertEqual(merged, expected)

    def test_merge_bins_works_if_no_bins_are_provided(self):
        self.assertRaises(AgeBin.NotMergeable, AgeBin.merge_bins, bins=[])

    def test_merge_bins_works_properly(self):
        bins = [AgeBin(0,5), AgeBin(5, 10), AgeBin(10, 15)]
        merged = AgeBin.merge_bins(bins=bins)
        expected = AgeBin(0, 15)
        self.assertEqual(merged, expected)

    def test_merge_bins_raises_if_unmergeable(self):
        bins = [AgeBin(0, 5), AgeBin(6, 10)]
        self.assertRaises(AgeBin.NotMergeable, AgeBin.merge_bins, bins=bins)

    # can_upsample_bins tests

    def test_can_upsample_bins_works_with_unsorted_AgeBins(self):
        bins = [AgeBin(5, 10), AgeBin(0, 5), AgeBin(10, 15)]
        self.assertTrue(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(0, 15)))
        self.assertTrue(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(0, 10)))
        self.assertTrue(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(5, 15)))
        self.assertTrue(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(5, 10)))

    def test_can_upsample_bins_works_if_no_bins_are_provided(self):
        bins = []
        self.assertFalse(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(0, 5)))

    def test_test_can_upsample_bins_works_with_AgeBin_and_string(self):
        bins = [AgeBin(0, 5), '[5:::10)', AgeBin(10, 15)]
        self.assertTrue(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(5, 15)))

    def test_can_upsample_bins_raises_if_target_bin_edges_do_not_line_up(self):
        bins = [AgeBin(0,5), AgeBin(5, 10), AgeBin(10, 15)]
        self.assertTrue(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(0, 15)))

        # a variety of misalignments relative to stated bins
        self.assertFalse(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(0, 14)))
        self.assertFalse(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(5, 11)))
        self.assertFalse(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(4, 10)))
        self.assertFalse(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(1, 15)))

    def test_can_upample_bins_raises_if_target_bin_not_contained_by_bins(self):
        bins = [AgeBin(0,5), AgeBin(5, 10), AgeBin(10, 15)]
        self.assertFalse(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(0, 16)))
        self.assertFalse(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(-1, 15)))

        self.assertFalse(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(15, 99)))
        self.assertFalse(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(16, 99)))

        self.assertFalse(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(-5, 0)))
        self.assertFalse(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(-5, 1)))


    def test_can_upsample_bins_works_properly(self):
        bins = [AgeBin(0,5), AgeBin(5, 10), AgeBin(10, 15)]
        self.assertTrue(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(0, 15)))
        self.assertTrue(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(0, 10)))
        self.assertTrue(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(5, 10)))
        self.assertTrue(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(5, 15)))
        self.assertTrue(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(0, 5)))
        self.assertTrue(AgeBin.can_upsample_bins(bins=bins, target_bin=AgeBin(10, 15)))

if __name__ == '__main__':
    unittest.main()
