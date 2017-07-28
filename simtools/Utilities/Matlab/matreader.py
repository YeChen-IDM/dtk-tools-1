from pprint import pprint
import scipy.io
import sys

# class ParameterPointTable(object):
#     def __init__(self, header, points):
#         self.header = header
#         self.original_points = points
#         self.points = []
#         for point_index in range(len(points)):
#             point_hash = {}
#             for header_index in range(len(header)):
#                 point_hash[ header[header_index] ] = points[point_index][header_index]
#             self.points.append(point_hash)

def read_mat_points_file(filename, header_key='jp', points_key='vals'):
    mat = scipy.io.loadmat(filename, squeeze_me=True, struct_as_record=False)
    points = mat[points_key]
    header = mat[header_key][0]
    #return cls(header=header, points=points)
    return header.tolist(), points.tolist()

# if __name__ == '__main__':
#     filename = sys.argv[1]
#     # tpis = ParameterPointTable.read_mat_points_file(filename)
#     #
#     # print('original data header\n---\n%s\n' % '\n'.join(tpis.header))
#     # print('original data format:\n---\n%s\n' % '\n'.join(map(lambda x: str(x), tpis.original_points)))
#     # print('constructed structure:\n---\n')
#     # pprint(tpis.points)
#     header, points = ParameterPointTable.read_mat_points_file(filename)
#