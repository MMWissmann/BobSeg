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
        if len(self.indices.shape)==2:
            triangles1=self.triangle_list(self.indices)
            triangles2=triangles1
        elif len(self.indices.shape)==len(self.vertices.shape):
            triangles1=self.triangle_list(self.indices[surface])
            triangles2=self.triangle_list(self.indices[surface+1])
        else: print('Array of indices should either be 2-dimensional or %s-dimensional'%len(vertices.shape))
        
        edge_points=self.find_edge_points(surface, triangles1)
        edge_points=np.stack((edge_points,self.find_edge_points(surface+1,triangles2)))
        
        vertices, indices = self.find_edge_neighbors(edge_points)
                
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
            t1=t.tolist()
            index=t1.index(vertex_index)
            b=vert[t[index]]
            tt=np.delete(t,index)
            a=vert[tt[0]]
            c=vert[tt[1]]
            
            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            deg_angle += np.degrees(angle)

        return deg_angle
                
    def find_edge_neighbors(self,edge_points):
        points_surf_1 = edge_points[0]
        points_surf_2 = edge_points[1]
        
        if len(points_surf_1) >= len(points_surf_2):
            more_edge_points = points_surf_1
            less_edge_points = points_surf_2
        elif len(points_surf_2) > len(points_surf_1):
            more_edge_points = points_surf_2
            less_edge_points = points_surf_1        
        else:
            print('Error in length of edge_points')  
        
        edge_points_sorted=[]
        edge_points_sorted.append((more_edge_points[0]))
        potential_neighbors=np.delete(more_edge_points,0,0)
        
        while len(edge_points_sorted)<len(more_edge_points):
            n,i=self.closest_node(edge_points_sorted[-1],potential_neighbors)
            edge_points_sorted.append((n))
            potential_neighbors=np.delete(potential_neighbors,i,0)
   
        neighbor=[]
    
        print(edge_points_sorted)

        for p in edge_points_sorted:
            node,index=self.closest_node(p,less_edge_points)
            neighbor.append((node))
            
        neighbor=np.array(neighbor)
        
        assert len(neighbor)==len(edge_points_sorted)
        
        vertices=np.append([edge_points_sorted],[neighbor],1)
        vertices=vertices[0]
        print('shape vert', vertices.shape)
        
        k=0
        k2=len(edge_points_sorted)
        indices=[]
        for p in edge_points_sorted:
            if k==k2-1:
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
            else:
                if np.array_equal(neighbor[k],neighbor[k+1]):
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
            k+=1

        return vertices, indices

    def closest_node(self, node, nodes):
        closest_index = distance.cdist([node], nodes).argmin()
        return nodes[closest_index], closest_index
        
        
        
        
        
        
        