import numpy as np
import math
from scipy.spatial import distance


class StitchSurfaces:
    '''
    Contains operations to 
        - stitch together meshes lying on top of each other, without contact: stitch()
        - stitching together meshes at the interfaces where the meshes touch: join()
    '''
    
    def __init__(self,vertices,indices,neighbors):
        '''
        vertices    :  Array with shape (n,x,3), where n is at least 2 (for 2 surfaces)
        indices     :  Indicating, which respective 3 vertices to use for triangulation,
                        either of shape (n,y,3) with same n as in vertices, or (y,3), which means same indices for all surfaces
        neighbors.  :  Adjacency list of arrays, containing indices of neighboring vertices for every vertex in list
        '''
        
        assert vertices.shape[0]>1
        
        self.indices=indices
        self.vertices=vertices
        self.neighbors=neighbors
            
    def stitch(self,surface):
        '''
        creating triangulation vertices and indices for a mesh connecting surface and surface+1 along their edges
        '''        
        if len(self.indices.shape)==2:   #in this case the indices in self.indices are valid for both/all surfaces
            triangles1=self.triangle_list(self.indices)
            triangles2=triangles1
        elif len(self.indices.shape)==len(self.vertices.shape):  #here every surface has own set of indices
            triangles1=self.triangle_list(self.indices[surface])
            triangles2=self.triangle_list(self.indices[surface+1])
        else: print('Array of indices should either be 2-dimensional or %s-dimensional'%len(vertices.shape))
        
        edge_points=self.find_edge_points(surface, triangles1) #edge_points are the vertices on edge of surface 
        edge_points2=self.find_edge_points(surface+1,triangles2) #edge-vertices of surface+1
        
        vertices, indices = self.find_edge_neighbors(edge_points,edge_points2) 
                
        return np.array(vertices), np.array(indices)
    
    def triangle_list(self,indices):
        '''
        Returns a list that assigns for every vertex-index the indices of the related triangles
          e.g. triangles[k] gives an 3,n-dim array of all the n triangles that k is part of
        '''        
        triangles = {}

        for point in range(max(len(self.vertices[0]),len(self.vertices[1]))):
            triangles[point] = []

        for simplex in indices:
            triangles[simplex[0]] += [simplex]
            triangles[simplex[1]] += [simplex]
            triangles[simplex[2]] += [simplex]
            
        return triangles
    
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
        
        
        
        
        
        
        