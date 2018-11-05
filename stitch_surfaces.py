import numpy as np
import math
from scipy.spatial import distance
import trimesh
from trimesh import grouping, intersections, geometry, transformations


class StitchSurfaces:
    '''
    Contains operations to 
        - stitch together meshes lying on top of each other, without contact: stitch()
        - stitching together meshes at the interfaces where the meshes touch: connect()
    '''
    
    def __init__(self,vertices,indices):
        '''
        vertices    :  Array with shape (n,x1,3), where n is at least 2 (for 2 surfaces), x1: num vertices
        indices     :  Indicating, which respective 3 vertices to use for triangulation,
                        either of shape (n,x2,3) with same n as in vertices, or (x2,3), which means same indices for all surfaces
                        x2: num indices per surface n
        '''
        
        assert vertices.shape[0]>1
        
        self.indices=indices
        self.vertices=vertices
            
    def stitch(self,surface):
        '''
        creating triangulation vertices and indices for a mesh connecting surface1 and surface2 along their edges
            surface= array [surface1,surface2]
        '''        
        assert len(surface)==2
        surface1=surface[0]
        surface2=surface[1]
        if len(self.indices.shape)==2:   #in this case the indices in self.indices are valid for both/all surfaces
            triangles1=self.triangle_list(self.indices)
            triangles2=triangles1
        elif len(self.indices.shape)==len(self.vertices.shape):  #here every surface has own set of indices
            triangles1=self.triangle_list(self.indices[surface1])
            triangles2=self.triangle_list(self.indices[surface2])
        else: print('Array of indices should either be 2-dimensional or %s-dimensional'%len(vertices.shape))
        
        edge_points=self.find_edge_points(surface1, triangles1) #edge_points are the vertices on edge of surface 
        edge_points2=self.find_edge_points(surface2,triangles2) #edge-vertices of surface+1
        
        vertices, indices = self.find_edge_neighbors(edge_points,edge_points2) 
                
        return np.array(vertices), np.array(indices)
    
    def connect(self,surface):
        '''
        Creating new vertices along the line of intersection of two meshes, to connect there 
        and delete the now unnecessary vertices beyond line of intersection
        surface gives the indication which set of vertices & indices to take
        '''
        assert len(surface)==2
        surface1=surface[0]
        surface2=surface[1]
        
        triangles1=self.triangle_list(self.indices[surface1])
        triangles2=self.triangle_list(self.indices[surface2])
        
        try: 
            intersec_points_1, intersec_points_2 = self.find_intersection(self.vertices[surface1],self.vertices[surface2])
            #intersec_points have the form [[index],[z,y,x]], where index is the index of the vertex [z,y,x] in self.vertices
            #here and in the following numbers 1 and 2 correspond to set of vertices of surface1 and surface2
        except: return None
        else:            
            tris1=self.find_inter_tris(intersec_points_1,triangles1)
            tris2=self.find_inter_tris(intersec_points_2,triangles2)
            #tris: indices of the triangles living in the intersection area
            
            self.create_new_vertices(tris1,tris2)
        
            connecting_tri()

            vertices,indices=full_tri()

            return np.array(vertices), np.array(indices)
            
    def triangle_list(self,vertices):
        '''
        Returns a list that assigns for every vertex-index the indices of the related triangles
          e.g. triangles[k] gives an 3,n-dim array of all the n triangles that k is part of
        '''        
        triangles = {}

        for point in range(max(len(self.vertices[0]),len(self.vertices[1]))):
            triangles[point] = []

        for simplex in vertices:
            triangles[simplex[0]] += [simplex]
            triangles[simplex[1]] += [simplex]
            triangles[simplex[2]] += [simplex]
            
        return triangles
    
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
                inds=self.indices
            elif len(self.indices.shape)>2:
                inds=self.indices[surface]

            for section in num_slices:
                new_origin = plane_orig + (plane_normal * section)
                new_dots = (vertex_dots - section)[inds]
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
        signs = np.zeros(self.indices.shape, dtype=np.int8)
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
             
        if len(self.indices.shape)==2:
            inds=self.indices
            sig=signs
        elif len(self.indices.shape)>2:
            inds=self.indices[surface]
            sig=signs[surface]
            
        if len(self.vertices.shape)==2:
            verts=self.vertices
        elif len(self.vertices.shape)>2:
            verts=self.vertices[surface]

        lines.append(np.vstack([h(sig[c],
                             inds[c],
                             verts)
                           for c, h in zip(cases, handlers)]))
        face_index = np.hstack([np.nonzero(c)[0] for c in cases])
        return lines, face_index
    

    '''
    Functions only for stitch()
    '''
    
    
    def find_edge_points(self,surface,triangles):
        '''
        Selecting all the vertices where the sum of all angles around the vertex in all related triangles is smaller 360
        Returns these vertices (all the vertices in surface that are on the edge of the surface)
        '''
        edge_points=[]
        
        k=0
        for v in self.vertices[surface]:
            angle=self.calculate_angle_around(k,triangles[k],surface)
            if not int(angle)==360:
                edge_points.append((v))
            k+=1
        
        return np.array(edge_points)
        
    def calculate_angle_around(self,vertex_index,triangles,surface):
        '''
        Calculates the angle around vertex with vertex_index in every triangle in triangles and adds these angles together
        '''        
        vert=np.delete(self.vertices[surface],2,1)
        
        deg_angle=0
        
        for t in triangles:
            #finding out, which of the three entries in t belongs to the considered vertex and naming it b
            t1=t.tolist()
            index=t1.index(vertex_index)
            b=vert[t[index]] #vertex
            tt=np.delete(t,index)
            a=vert[tt[0]] #a and c are the two other vertices in the triangle t
            c=vert[tt[1]]
            
            ba = a - b #creating vectors along the two sides of t, that start in b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle) #angle in radian
            deg_angle += np.degrees(angle)  #adding angle in degrees to the other angles around the regarded vertex

        return deg_angle
                
    def find_edge_neighbors(self,edge_points,edge_points2):
        '''
        Returns two lists: 1. list of all vertices at the edges of the two surfaces
                           2. list of indices needed for creating the triangulation
        Parameters:
        edge_points : (n1,3)-dim array containing the n1 vertices on the edge of surface1
        edge_points2: (n2,3)-dim array containing the n2 vertices on the edge of surface2
        '''
        
        #first: finds the surface with more edge-vertices and defines list of these vertices as more_edge_points
        if len(edge_points) >= len(edge_points2):
            more_edge_points = edge_points
            less_edge_points = edge_points2
        elif len(edge_points2) > len(edge_points):
            more_edge_points = edge_points2
            less_edge_points = edge_points      
        else:
            print('Error in length of edge_points')  
        
        #second: sorts the list of edge-vertices in more_edge_points according to order along the surface edge
        edge_points_sorted=[]
        edge_points_sorted.append((more_edge_points[0]))
        potential_neighbors=np.delete(more_edge_points,0,0)
        
        while len(edge_points_sorted)<len(more_edge_points):
            n,i=self.closest_node(edge_points_sorted[-1],potential_neighbors)
            edge_points_sorted.append((n))
            potential_neighbors=np.delete(potential_neighbors,i,0)
       
        #third: finds for every element in more_edge_points the nearest vertex in less_edge_points, saves in neighbor
        #that means, nearest vertex on other surface to vertex edge_points_sorted[k] is: neighbor[k]
        neighbor=[]

        for p in edge_points_sorted:
            node,index=self.closest_node(p,less_edge_points)
            neighbor.append((node))
            
        neighbor=np.array(neighbor)
        
        assert len(neighbor)==len(edge_points_sorted)
        
        #fourth: creates complete list of all vertices, neighbor to vertices[k] is now vertices[k2+k]
        k2=len(edge_points_sorted)
        vertices=np.append([edge_points_sorted],[neighbor],1) 
        vertices=vertices[0]
        
        #fifth: creates list of indices for triangulation
        k=0
        indices=[]
        for p in edge_points_sorted:
            if not k==k2-1:    
                if np.array_equal(neighbor[k],neighbor[k+1]):
                    #triangle consisting of p, the next vertex in edge_points_sorted and the neighbor of p
                    indices.append((k,k+1,k2+k))
                else:
                    check1=np.append([edge_points_sorted[k]],[neighbor[k+1]],0)
                    check2=np.append([edge_points_sorted[k+1]],[neighbor[k]],0)
                    if distance.pdist(check1)<=distance.pdist(check2):
                        indices.append((k,k+1,k2+k+1))
                        indices.append((k,k2+k,k2+k+1))
                    else:
                        indices.append((k,k+1,k2+k))
                        indices.append((k+1,k2+k,k2+k+1))
            else:
                #this is only the case for last element in edge_points_sorted: connects to the first element to get closed shape
                if np.array_equal(neighbor[k],neighbor[0]):
                    indices.append((k,0,k2+k))
                else:
                    check1=np.append([edge_points_sorted[k]],[neighbor[0]],0)
                    check2=np.append([edge_points_sorted[0]],[neighbor[k]],0)
                    if distance.pdist(check1)<=distance.pdist(check2):
                        indices.append((k,0,k2+0))
                        indices.append((k,k2+k,k2+0))
                    else:
                        indices.append((k,0,k2+k))
                        indices.append((0,k2+k,k2+0))                
            k+=1

        return vertices, indices

    def closest_node(self, node, nodes):
        closest_index = distance.cdist([node], nodes).argmin()
        return nodes[closest_index], closest_index
        
        
        
    '''
    Functions only for connect()
    '''
    
    
        
    def find_intersection(self, vertices1, vertices2):
        
        #still wrong.. 0,1,2 should give the z,y,x parts of the arrays
        max1=[max(vertices1[:,0])]
        max2=[max(vertices2[:,0])]
        min1=[min(vertices1[:,0])]
        min2=[min(vertices2[:,0])]
        
        intersection1x=[]
        intersection2x=[]
        l=0
        m=0
        if min1<max2<max1: 
            for vx in vertices1:
                if vx[2]<max2:
                    intersection1x.append(([l],vx))
                l+=1
            for vx in vertices2:
                if vx[2]>min1:
                    intersection2x.append(([m],vx))
                m+=1
        elif min2<max1<min2:
            for vx in vertices2:
                if vx[2]<max1:
                    intersection2x.append(([m],vx))
                m+=1
            for vx in vertices1:
                if vx[2]>min2:
                    intersection1x.append(([l],vx))
                l+=1
        else: print('No overlap of surfaces in x')
            
        intersection1xy=[]
        intersection2xy=[]
        
        if not intersection1x==[]:
            
            intersection1x=np.array(intersection1x)[0]
            intersection2x=np.array(intersection2x)[0]

            max1=max(np.array(intersection1x[1])[:,1])
            max2=max(np.array(intersection2x[1])[:,1])
            min1=min(np.array(intersection1x[1])[:,1])
            min2=min(np.array(intersection2x[1])[:,1])
            
            if min1<max2<max1: 
                for vy in intersection1x:
                    if vy[1][1]<max2:
                        intersection1xy.append((vy))
                for vy in intersection2x:
                    if vy[1][1]>min1:
                        intersection2xy.append((vy))
            elif min2<max1<min2:
                for vy in intersection2x:
                    if vy[1][1]<max1:
                        intersection1xy.append((vy))
                for vy in intersection1x:
                    if vy[1][1]>min2:
                        intersection2xy.append((vy))
            else: print('No overlap of surfaces in y')
        
        intersection1xyz=[]
        intersection2xyz=[]
        
        if not intersection1xy==[]:
            
            intersection1xy=np.array(intersection1xy)[0]
            intersection2xy=np.array(intersection2xy)[0]

            max1=max(np.array(intersection1xy[1])[:,0])
            max2=max(np.array(intersection2xy[1])[:,0])
            min1=min(np.array(intersection1xy[1])[:,0])
            min2=min(np.array(intersection2xy[1])[:,0])
            
            if min1<max2<max1: 
                for vz in intersection1xy:
                    if vz[1][0]<max2:
                        intersection1xyz.append((vz))
                for vy in intersection2xy:
                    if vz[1][0]>min1:
                        intersection2xyz.append((vz))
            elif min2<max1<min2:
                for vz in intersection2xy:
                    if vz[1][0]<max1:
                        intersection1xyz.append((vz))
                for vz in intersection1xy:
                    if vz[1][0]>min2:
                        intersection2xyz.append((vz))
            else: print('No overlap of surfaces in z')
            
        if not intersection1xyz==[]:
            intersection1=np.array(intersection1xyz)[0]
            intersection2=np.array(intersection2xyz)[0]
            
            return intersection1,intersection2
        
        else: return None

    def find_inter_tris(self,inter_verts,indices):
        inter_index=inter_verts[0]
        
        tris=[]
        
        for i in inter_index:            
            tris.append(indices[i])
        
        return np.array(tris)
        
  #  def create_new _vertices(self,indices1, indices2):
        