import openpyxl.utils


def read_block(ws, range):
    """
    Reads data from the specified range. Result is a list of data by row, each of those being a list (by col).
    Converts all '' strings to None.
    :param ws: an openpyxl worksheet object
    :param range: an openpyxl range object
    :return: data from the requested 2D worksheet range, in row-major order (list of lists)
    """
    data = []
    min_col, min_row, max_col, max_row = openpyxl.utils.range_boundaries(range.cells)
    for row in ws.iter_rows(min_col=min_col, max_col=max_col, min_row=min_row, max_row=max_row):
        data.append([item.value if item.value != '' else None for item in row])
    return data


def read_list(ws, range):
    """
    Read in a linear 1xN or Nx1 range of cells from an excel spreadsheet
    :param ws: an openpyxl worksheet object
    :param range: an openpyxl range object
    :return: data from the requested row/column as a list of items
    """
    min_col, min_row, max_col, max_row = openpyxl.utils.range_boundaries(range.cells)
    is_row = min_row == max_row
    is_col = min_col == max_col

    if not (is_row or is_col):
        raise Exception('Not a linear range: %s' % range)

    data = read_block(ws, range)
    if is_row:
        data = data[0]
    else:
        data = [item[0] for item in data]
    return data


class DefinedName(object):
    def __init__(self, openpyxl_defined_range):
        self.opdr = openpyxl_defined_range

        self.name = self.opdr.name
        self.sheet, self.cells = self.opdr.attr_text.split('!')
        self.scope = 'workbook' if self.opdr.localSheetId is None else self.sheet.replace('\'','')

    @classmethod
    def load_from_workbook(cls, wb):
        defined_names = wb.defined_names.definedName
        lookup_dict = {}
        for defined_name in defined_names:
            dn = cls(defined_name)
            lookup_dict[dn.scope] = lookup_dict.get(dn.scope, None) or dict()
            lookup_dict[dn.scope][dn.name] = dn
        return lookup_dict
