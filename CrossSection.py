import numpy as np
from trimesh import grouping, intersections, geometry, transformations

class CrossSection:
    '''
    Code basically taken from the trimesh package,
    small modifications, simplifications for this specific purpose
    '''
    
    def __init__(self,vertices,indices):
        '''
        vertices    :  Array with shape (n,x1,3), where n is number of surfaces, x1: num vertices
        indices     :  Indicating, which respective 3 vertices to use for triangulation,
                        either of shape (n,x2,3) with same n as in vertices, or (x2,3), which means same indices for all surfaces
                        x2: num indices per surface n
        '''
        vertices = np.asanyarray(vertices,dtype=np.float64)    
        indices  = np.asanyarray(indices,dtype=np.int)    
        vertices,indices= np.squeeze(vertices),np.squeeze(indices)
        assert vertices.shape[0]>1
        self.vertices=vertices
        self.indices=indices
    
    def get_multisections(self,plane_orig,plane_normal,num_slices):
        plane_origin = np.asanyarray(plane_orig,dtype=np.float64)
        num_slices = np.asanyarray(num_slices, dtype=np.float64)
        
        # store results
        segments = {}
        
        for surface in range(len(self.vertices)):
            
            segments[surface]=[]
            
            # dot product of each vertex with the plane normal indexed by face
            # so for each face the dot product of each vertex is a row
            # shape is the same as self.indices (n,3)
            
            if len(self.vertices.shape)==3:
                vertex_dots = np.dot(plane_normal,(self.vertices[surface] - plane_origin).T)
            else:
                vertex_dots = np.dot(plane_normal,(self.vertices - plane_origin).T)

            i=0
            
            if len(self.indices.shape)==2:
                self.inds=self.indices
            elif len(self.indices.shape)>2:
                self.inds=self.indices[surface]
            else: print('Sth wrong with indices shape (should not be one-dimensional array)')

            for section in num_slices:
                new_origin = plane_orig + (plane_normal * section)
                new_dots = (vertex_dots - section)[self.inds]
                cross_section, indices = self.get_section(plane_orig=new_origin,plane_normal=plane_normal,
                                                          dots=new_dots,surface=surface)

                segments[surface].append(cross_section)

        return segments
            
    def get_section(self,plane_orig,plane_normal,dots,surface):
        
        def triangle_cases(signs):
            """
            Figure out which faces correspond to which intersection
            case from the signs of the dot product of each vertex.
            Does this by bitbang each row of signs into an 8 bit
            integer.
            code : signs      : intersects
            0    : [-1 -1 -1] : No
            2    : [-1 -1  0] : No
            4    : [-1 -1  1] : Yes; 2 on one side, 1 on the other
            6    : [-1  0  0] : Yes; one edge fully on plane
            8    : [-1  0  1] : Yes; one vertex on plane, 2 on different sides
            12   : [-1  1  1] : Yes; 2 on one side, 1 on the other
            14   : [0 0 0]    : No (on plane fully)
            16   : [0 0 1]    : Yes; one edge fully on plane
            20   : [0 1 1]    : No
            28   : [1 1 1]    : No
            Parameters
            ----------
            signs: (n,3) int, all values are -1,0, or 1
                   Each row contains the dot product of all three vertices
                   in a face with respect to the plane
            Returns
            ---------
            basic:      (n,) bool, which faces are in the basic intersection case
            one_vertex: (n,) bool, which faces are in the one vertex case
            one_edge:   (n,) bool, which faces are in the one edge case
            """

            signs_sorted = np.sort(signs, axis=1)
            coded = np.zeros(len(signs_sorted), dtype=np.int8) + 14
            for i in range(3):
                coded += signs_sorted[:, i] << 3 - i

            # one edge fully on the plane
            # note that we are only accepting *one* of the on- edge cases,
            # where the other vertex has a positive dot product (16) instead
            # of both on- edge cases ([6,16])
            # this is so that for regions that are co-planar with the the section plane
            # we don't end up with an invalid boundary
            key = np.zeros(29, dtype=np.bool)
            key[16] = True
            one_edge = key[coded]

            # one vertex on plane, other two on different sides
            key[:] = False
            key[8] = True
            one_vertex = key[coded]

            # one vertex on one side of the plane, two on the other
            key[:] = False
            key[[4, 12]] = True
            basic = key[coded]

            return basic, one_vertex, one_edge

        def handle_on_vertex(signs, faces, vertices):
            # case where one vertex is on plane, two are on different sides
            vertex_plane = faces[signs == 0]
            edge_thru = faces[signs != 0].reshape((-1, 2))
            point_intersect, valid = intersections.plane_lines(plane_orig,
                                                 plane_normal,
                                                 vertices[edge_thru.T],
                                                 line_segments=False)
            lines = np.column_stack((vertices[vertex_plane[valid]],
                                     point_intersect)).reshape((-1, 2, 3))
            return lines

        def handle_on_edge(signs, faces, vertices):
            # case where two vertices are on the plane and one is off
            edges = faces[signs == 0].reshape((-1, 2))
            points = vertices[edges]
            return points

        def handle_basic(signs, faces, vertices):
            # case where one vertex is on one side and two are on the other
            unique_element = grouping.unique_value_in_row(
                signs, unique=[-1, 1])
            edges = np.column_stack(
                (faces[unique_element],
                 faces[np.roll(unique_element, 1, axis=1)],
                 faces[unique_element],
                 faces[np.roll(unique_element, 2, axis=1)])).reshape(
                (-1, 2))
            
            intersec, valid = intersections.plane_lines(plane_orig,
                                               plane_normal,
                                               vertices[edges.T],
                                               line_segments=False)
            # since the data has been pre- culled, any invalid intersections at all
            # means the culling was done incorrectly and thus things are
            # mega-fucked
            assert valid.all()
            intersec=intersec[valid]
            return np.array(intersec).reshape((-1, 2, 3))
        
        # sign of the dot product is -1, 0, or 1
        # shape is the same as self.indices (n,3)
        signs = np.zeros(self.inds.shape, dtype=np.int8)
        signs[dots < -1e-5] = -1
        signs[dots > 1e-5] = 1
        
        # figure out which triangles are in the cross section,
        # and which of the three intersection cases they are in
        cases = triangle_cases(signs)
        
        # handlers for each case
        handlers = (handle_basic,
                    handle_on_vertex,
                    handle_on_edge)

        # the (m, 2, 3) line segments
        lines=[]

        if len(self.vertices.shape)==2:
            verts=self.vertices
        elif len(self.vertices.shape)>2:
            verts=self.vertices[surface]
        lines.append(np.vstack([h(signs[c],
                             self.inds[c],
                             verts)
                           for c, h in zip(cases, handlers)]))
        face_index = np.hstack([np.nonzero(c)[0] for c in cases])
        return lines, face_index