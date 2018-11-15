import numpy as np
import math
from scipy.spatial import distance
import trimesh
from trimesh import grouping, intersections, geometry, transformations


class StitchSurfaces:
    '''
    Contains operations to 
        - stitch together meshes lying on top of each other, without contact: stitch()
        - stitching together meshes at the interfaces where the meshes touch: connect() TODO
    '''
    
    def __init__(self,vertices,indices):
        '''
        vertices    :  Array with shape (n,x1,3), or list with n entries of (xi,3) where n is at least 2 (for 2 surfaces), x1: num vertices
        indices     :  Indicating, which respective 3 vertices to use for triangulation,
                        either of shape (n,x2,3) (or list respectively) with same n as in vertices, or (x2,3), which means same indices for all surfaces
                        x2: num indices per surface n
        '''
        
        self.indices=indices
        self.vertices=vertices
            
    def stitch(self,surface):
        '''
        creating triangulation vertices and indices for a mesh connecting surface1 and surface2 along their edges
            surface= array [surface1,surface2]
        '''        
        assert self.vertices.shape[0]>1
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
        
        vertices, indices = self.create_stitching(edge_points,edge_points2) 
                
        return np.array(vertices), np.array(indices)
        
    def connect(self, main_surface, connect_surface):
        
        assert len(self.indices[connect_surface].shape) ==2 and len(self.vertices[connect_surface].shape) ==2
        assert len(self.indices[main_surface].shape)==3 or len(self.indices[main_surface].shape)==2
        assert len(self.vertices[main_surface].shape)==3 or len(self.vertices[main_surface].shape)==2
        
        self.main_surface = main_surface
        self.connect_surface = connect_surface
          
        connect_triangles = self.triangle_list(self.indices[connect_surface]) 
        edge_points = self.find_edge_points(connect_surface,connect_triangles,transform=True)
        
        vertices,indices = self.create_connection_with(edge_points)
        
        return np.array(vertices), np.array(indices)
 
            
    def triangle_list(self,vertices):
        '''
        Returns a list that assigns for every vertex-index the indices of the related triangles
          e.g. triangles[k] gives an 3,n-dim array of all the n triangles that k is part of
        '''        
        triangles = {}
        
        for point in range(len(vertices)):
            triangles[point] = []

        for simplex in vertices:
            triangles[simplex[0]] += [simplex]
            triangles[simplex[1]] += [simplex]
            triangles[simplex[2]] += [simplex]
            
        return triangles
    
    
    '''
    Functions
    '''
    
    
    def find_edge_points(self,surface,triangles,transform=False):
        '''
        Selecting all the vertices where the sum of all angles around the vertex in all related triangles is smaller 360
        Returns these vertices (all the vertices in surface that are on the edge of the surface)
        '''
        if transform and (surface==1 or surface==2):
            axis=abs(surface-2)        #from [012] --> [210]
            self.vertices[surface][:,[2,axis]] = self.vertices[surface][:,[axis,2]]
        
        edge_points=[]
        
        k=0
        for v in self.vertices[surface]:
            angle=self.calculate_angle_around(k,triangles[k],surface)
            if not int(angle)==360:
                edge_points.append((v))
            k+=1
            
        edge_points = np.array(edge_points)
        
        if transform and (surface==1 or surface==2):        #from [210] --> [012]
            self.vertices[surface][:,[2,axis]] = self.vertices[surface][:,[axis,2]]
            edge_points[:,[2,axis]] = edge_points[:,[axis,2]]
        
        return edge_points
        
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
                
    def create_stitching(self,edge_points,edge_points2):
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

        edge_points_sorted = self.sort_edge_points(more_edge_points)
        
        #third: finds for every element in edge_points_sorted (former more_edge_points) the nearest vertex in less_edge_points,   
        #saves in neighbor
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
        
        indices = self.create_indices(edge_points_sorted,neighbor)
        
        return vertices, indices
    
    def create_connection_with(self,edge_points):
        
        edge_points_sorted = self.sort_edge_points(edge_points)
        
        if len(self.vertices[self.main_surface].shape)==3:
            total_num_verts=self.vertices[self.main_surface].shape[0]*self.vertices[self.main_surface].shape[1]
            main_vertices = self.vertices[self.main_surface].reshape(total_num_verts,3)
        else:
            main_vertices=self.vertices[self.main_surface]
            
        neighbor=[]

        for p in edge_points_sorted:
            node,index=self.closest_node(p,main_vertices)
            neighbor.append((node))
            
        neighbor=np.array(neighbor)
        
        assert len(neighbor)==len(edge_points_sorted)
        
        vertices=np.append([edge_points_sorted],[neighbor],1) 
        vertices=vertices[0]
        
        indices=self.create_indices(edge_points_sorted,neighbor)
        
        return vertices, indices

    def closest_node(self, node, nodes):
        closest_index = distance.cdist([node], nodes).argmin()
        return nodes[closest_index], closest_index        
      
    def sort_edge_points(self,points):
        edge_points_sorted=[]
        edge_points_sorted.append((points[0]))
        potential_neighbors=np.delete(points,0,0)
        
        while len(edge_points_sorted)<len(points):
            n,i=self.closest_node(edge_points_sorted[-1],potential_neighbors)
            edge_points_sorted.append((n))
            potential_neighbors=np.delete(potential_neighbors,i,0)
            
        return edge_points_sorted
    
    def create_indices(self,edge_points_sorted,neighbor):
        k2=len(edge_points_sorted)
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
            
        return indices





    '''
    Draft for more advanced union of meshes
    bad, can probably be deleted
    '''


    def connect2(self,surface):
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
        