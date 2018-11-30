import numpy as np
import math
from scipy.spatial import distance, Delaunay
from trimesh import grouping, intersections, geometry, transformations, boolean, graph
#import pymesh


class StitchSurfaces:
    '''
    Contains operations to 
        - stitch together meshes lying on top of each other, without contact: stitch()
        - stitching together meshes at the interfaces where the meshes touch: connect() TODO
        - Making closed, watertight 3D meshes out of surface meshes BUGGY
        - Perform Boolean operations on these watertight meshes using trimesh(blender,openscad), also possible when commenting out/in respective parts: using pymesh(cork,igl,carve,cgal)
    '''
    
    def __init__(self,vertices,indices,image_shape=None):
        '''
        vertices    :  Array with shape (n,x1,3), or list with n entries of (xi,3) where n is at least 2 (for 2 surfaces), 
                                                x1: num vertices
        indices     :  Indicating, which respective 3 vertices to use for triangulation,
                        either of shape (n,x2,3) (or list respectively) with same n as in vertices, or (x2,3), 
                                                which means same indices for all surfaces
                        x2: num indices per surface n
        '''
        
        self.indices=indices
        self.vertices=vertices
        self.image_shape=image_shape
            
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
        
            
    def triangle_list(self,vertex_indices):
        '''
        Returns a list that assigns for every vertex-index the indices of the related triangles
          e.g. triangles[k] gives an 3,n-dim array of all the n triangles that k is part of
        '''        
        triangles = {}
        
        maximum=0
        for tri in vertex_indices:
            for one_tri in tri:
                maximum=max(maximum,one_tri)
        
        print(maximum)
                
        for point in range(maximum+1):
            triangles[point] = []

        for simplex in vertex_indices:
            triangles[simplex[0]] += [simplex]
            triangles[simplex[1]] += [simplex]
            triangles[simplex[2]] += [simplex]
            
        return triangles
    
    def remove_verts_beyond(self, border_vertices, main_vertices, main_indices, direction, searching_direction):
       
        relevant_border_vertices=[]
        for border_vertex in main_vertices:
            relevant_border_vertices.append((border_vertex))
                
        for vert in main_vertices:
            self.find_nearest(relevant_border_vertices,vert,searching_direction)
            
        
    def find_nearest(array, value,searching_direction):
        array = np.asarray(array)
        new_array=array[:,searching_direction]
        value=value[searching_direction]
        idx = (np.abs(new_array - value)).argmin()
        return array[idx]   
    
    
    '''
    Functions
    '''
    
    def find_indices(self,value, my_list,d):
        index=[]
        while value in my_list[:,d]:
            value_index = np.where(my_list[:,d]==value)
            index.append(value_index)
            my_list=np.delete(my_list,value_index,0)
    #             my_list[value_index]=[0,0,0]

        return index, my_list
    
    
    def find_edge_points(self,surface,triangles,transform=False):
        '''
        Selecting all the vertices where the sum of all angles around the vertex in all related triangles is smaller 360
        Returns these vertices (all the vertices in surface that are on the edge of the surface)
        '''
        #if x- or y- method: change z-coordinate back to x or y...
        if transform and (surface==1 or surface==2):
            axis=abs(surface-2)        #from [012] --> [210]
            self.vertices[surface][:,[2,axis]] = self.vertices[surface][:,[axis,2]]
        
        edge_points=[]
        
        for k,v in enumerate(self.vertices[surface]):
            angle=self.calculate_angle_around(k,triangles[k],self.vertices[surface])
            if not int(angle)==360:
                edge_points.append((v))
            
        edge_points = np.array(edge_points)
        
        #...and change back to z
        if transform and (surface==1 or surface==2):        #from [210] --> [012]
            self.vertices[surface][:,[2,axis]] = self.vertices[surface][:,[axis,2]]
            edge_points[:,[2,axis]] = edge_points[:,[axis,2]]
        
        return edge_points
        
    def calculate_angle_around(self,vertex_index,triangles,vertices,axis=0):
        '''
        Calculates the angle around vertex with vertex_index in every triangle in triangles and adds these angles together
        '''        
        vert=np.delete(vertices,2,1)
        
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

    def closest_node(self, node, nodes):
        #Returns element of nodes (and its index) that has minimal distance to node
        closest_index = distance.cdist([node], nodes).argmin()
        return nodes[closest_index], closest_index        
      
    def sort_edge_points(self,points, adjacency_list=None,vertices=None):
        '''
        Sort points, so that ajacent elements in points represent adjacent nodes 
        last node should be connected to first node
        '''
        edge_points_sorted=[]
        edge_points_sorted.append((points[0]))
        potential_neighbors=np.delete(points,0,0)
        
        if adjacency_list is None:
            while len(edge_points_sorted)<len(points):
                n,i=self.closest_node(edge_points_sorted[-1],potential_neighbors)
                edge_points_sorted.append((n))
                potential_neighbors=np.delete(potential_neighbors,i,0)
        else:
            while len(edge_points_sorted)<len(points)-1:
                n,i=self.find_next_vert(edge_points_sorted[-1],potential_neighbors,adjacency_list,vertices)
                edge_points_sorted.append((n))
                potential_neighbors=np.delete(potential_neighbors,i,0)
            edge_points_sorted.append((potential_neighbors[0]))
            assert len(edge_points_sorted)==len(points)
        
        return np.array(edge_points_sorted)
    
    def find_next_vert(self,point,array,adjacency,vertices):
        '''
        Goes from point to adjacent points, saves exactly one adjacent point in vertex, that is also an edge_point
        '''
        vertex =None
        index=None
        original_index=np.where([point.tolist()==v for v in vertices.tolist()])[0][0]
        for n in adjacency[original_index]:
            if vertices[n].tolist() in array.tolist():
                vertex = vertices[n]
                index = np.where([vertices[n].tolist()==v for v in array.tolist()])[0][0]
        
        assert not vertex is None
        assert not index is None
        
        return vertex, index
    
    def adjacency_list(self,points,indices):
        #produce adjacency list
        neighbors = {}
        
        for point in range(len(points)):
            neighbors[point] = []

        for simplex in indices:
            neighbors[simplex[0]] += [simplex[1],simplex[2]]
            neighbors[simplex[1]] += [simplex[2],simplex[0]]
            neighbors[simplex[2]] += [simplex[0],simplex[1]]
        
        #remove doubles and sort by value
        for a in range(len(neighbors)):
            neighbors[a]=list(set(neighbors[a]))
            neighbors[a]=np.sort(neighbors[a])
            
        return neighbors
    
    def create_indices(self,edge_points_sorted,neighbor):
        '''
        Returns indices that make triangulation to connect edge_points_sorted to neighbor
        '''
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
    CONSTRUCTION SITE: 
    Boolean operations, making closed volume meshes out of surfeace meshes and all stuff to put meshes together except stitch()
    '''
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
        
        neighbor=self.sort_edge_points
        neighbor=np.array(neighbor)
        
        assert len(neighbor)==len(edge_points_sorted)
        
        vertices=np.append([edge_points_sorted],[neighbor],1) 
        vertices=vertices[0]
        
        indices=self.create_indices(edge_points_sorted,neighbor)
        
        return vertices, indices, neighbor
    def connect(self, main_surface, connect_surface):
        
        assert len(self.indices[connect_surface].shape) ==2 and len(self.vertices[connect_surface].shape) ==2
        assert len(self.indices[main_surface].shape)==3 or len(self.indices[main_surface].shape)==2
        assert len(self.vertices[main_surface].shape)==3 or len(self.vertices[main_surface].shape)==2
        
        self.main_surface = main_surface
        self.connect_surface = connect_surface
          
        connect_triangles = self.triangle_list(self.indices[connect_surface]) 
        edge_points = self.find_edge_points(connect_surface,connect_triangles,transform=True)
        
        vertices,indices, neighbor = self.create_connection_with(edge_points)
        
        return np.array(vertices), np.array(indices), np.array(neighbor)
    
    def make_closed(self,parameters):
        '''
        Takes those self.vertices and self.indices that characterize surfaces
        Connects the surfaces with the base_graph, in order to get closed, watertight volume for boolean operations
        Returns vertices and indices 
            (self.vertices and self.indices + new created vertices at edge of base and indices connecting base and surface)
        '''
        dx=parameters[0]
        dy=parameters[1]
        vertices={}
        indices={}
        k=0
        print(self.vertices)
        for verts in self.vertices:
            print('verts',verts)
            vertices[verts]={}
            indices[verts]={}
            if len(self.vertices[verts]):
                if len(self.vertices[verts].shape)==3:
                    if len(self.indices[verts].shape)==3:
                        for surf in range(len(self.vertices[verts])):
                            vertices[verts][surf]=[]
                            indices[verts][surf]=[]
                            if surf==0:
                                v,i = self.add_zeros_and_close(self.vertices[verts][surf],self.indices[verts][surf],
                                                           verts,dx[verts],dy[verts])
                            else:
                                 v,i = self.add_zeros_and_close(self.vertices[verts][surf],self.indices[verts][surf],
                                                           verts,dx[verts],dy[verts],main=False)
                            vertices[verts][surf]=v
                            indices[verts][surf]=i
                    elif len(self.indices[verts].shape)==2:
                        for surf in range(len(self.vertices[verts])):
                            vertices[verts][surf]=[]
                            indices[verts][surf]=[]
                            if surf ==0:
                                v, i = self.add_zeros_and_close(self.vertices[verts][surf],
                                                            self.indices[verts],verts,dx[verts],dy[verts])
                            else:
                                v, i = self.add_zeros_and_close(self.vertices[verts][surf],
                                                            self.indices[verts],verts,dx[verts],dy[verts],main=False)
                            vertices[verts][surf]=v
                            indices[verts][surf]=i
                            print('surf',surf)
#                             print(vertices[verts])
                elif len(self.vertices[verts].shape)==2:
                    v, i = self.add_zeros_and_close(self.vertices[verts],self.indices[verts],
                                                    verts,dx[verts],dy[verts])
                    vertices[verts]=v
                    indices[verts]=i
            k+=1
                    
        return vertices,indices
                   
            
    def add_zeros_and_close(self,vertices,indices,transform,dx,dy,main=True):
        '''
        Takes a set of vertices and indices and the base_graph method: transform (either 0==z, 1==y, or 2==x)
        Returns original vertices plus additional vertices on base graph edge
            and original indices plus additional indices for 
                1) making a closed surface on base graph
                2) making a closed surface to connect the original one to the new created base surface
        '''
        edge_points=[]
        
        print('main',main)
        
        # changes order of coordinates in vertices
        # e.g. for a set of vertices created with y-base-graph-method: [x,y,z] --> [x,z,y]
        if transform==1 or transform == 2:
            axis=abs(transform-2)        #from [012] --> [210]
            vertices[:,[2,axis]] = vertices[:,[axis,2]]
        
#         print(transform)
        # list of all triangles that every vertex is part of
        triangles=self.triangle_list(indices)
        
        #find edge points of the surface, they will be part of new created triangles connecting to the bottom
        for k,v in enumerate(vertices):
            angle=self.calculate_angle_around(k,triangles[k],vertices,transform)
            if not int(angle)==360:
                edge_points.append((v))
        
        #sort these edge points, i.e. adjacent entries in edge_points will be neighbored vertices along the edge
        edge_points = np.array(edge_points)
        adjacency_list = self.adjacency_list(vertices, indices)
        edge_points = self.sort_edge_points(edge_points, adjacency_list, vertices)
        (minimum_x,minimum_y)=(np.amin(edge_points[:,0]),np.amin(edge_points[:,1]))
        (maximum_x,maximum_y)=(np.amax(edge_points[:,0]),np.amax(edge_points[:,1]))
        minimum=(minimum_x,minimum_y)
        maximum=(maximum_x,maximum_y)                               
        
        edge_points=self.edge_points_start_at_corner(edge_points,minimum,maximum)

        #creates ground_points, those are the edge point vertices, just with: edge_point [x1,x2,x3] --> [x1,x2,0]
        #creates ground indices for a closed triangulation to connect edge_points with ground_points
        #creates translation_list: gives for every ground_point index the respective edge_point index
        num_verts=len(vertices)
        ground_points=[]
        ground_index=[]
        translation_list=[]
        k=0
        print('edge points',edge_points)
        for index,point in enumerate(edge_points):
            original_index=np.where([point.tolist()==v for v in vertices.tolist()])[0][0]
            if not index==len(edge_points)-1:
                next_original_index=np.where([edge_points[index+1].tolist()==v for v in vertices.tolist()])[0][0]
            translation_list.append((original_index))
            if not main:
                ground_point=point[:]
                ground_point[2]=-1
                ground_points.append(ground_point.tolist())
                if not index==len(edge_points)-1:
                    ground_index.append([num_verts+index, original_index,    next_original_index])
                    ground_index.append([num_verts+index, num_verts+index+1, next_original_index])
                else:
                    ground_index.append([num_verts+index, original_index, translation_list[0]])
                    ground_index.append([num_verts+index, num_verts+0,    translation_list[0]])
            if main:
                ground_point=point[:]
                ground_point[2]=0
                ground, corner = self.create_ground_points(ground_point,minimum,maximum)
                if not index==len(edge_points)-1:
                    unimportant, next_corner = self.create_ground_points(edge_points[index+1],minimum,maximum)
                else:
                    unimportant, next_corner = self.create_ground_points(edge_points[0],minimum,maximum)
#                 print(ground,corner,next_corner)
#                 print(k)
#                 print(index, len(edge_points))
                if not (corner or next_corner):
                    if not index>=len(edge_points)-2:
                        ground_index.append([num_verts+index-k, original_index,    next_original_index])
                        ground_index.append([num_verts+index-k, num_verts+index-k+1, next_original_index])
#                         print('normal',[num_verts+index-k, original_index,    next_original_index])
#                         print([num_verts+index-k, num_verts+index-k+1, next_original_index])
                    elif index == len(edge_points)-1:
                        ground_index.append([num_verts+index-k, original_index, translation_list[0]])
                        ground_index.append([num_verts+index-k, num_verts+0,    translation_list[0]])
#                         print('normal ende',[num_verts+index-k, original_index, translation_list[0]])
#                         print([num_verts+index-k, num_verts+0,    translation_list[0]])
                    elif index == len(edge_points)-2:
                        ground_index.append([num_verts+index-k, original_index,    next_original_index])
                        ground_index.append([num_verts+index-k, num_verts+0,  next_original_index])
#                         print('normal fast ende',[num_verts+index-k, original_index,    next_original_index])
#                         print([num_verts+index-k, num_verts+0,  next_original_index])
                    ground_points.append(ground)
#                     print(ground)
                if corner or next_corner:
                    if not index>=len(edge_points)-1:
                        ground_index.append([num_verts+index-k, original_index,   next_original_index])
#                         print('corner',[num_verts+index-k, original_index,   next_original_index])
                        k+=1
                    elif index==len(edge_points)-1:
                        ground_index.append([num_verts+0, original_index, translation_list[0]])
#                         print('corner ende',[num_verts+0, original_index, translation_list[0]])
                
#         print(np.array(vertices).shape)
#         print(np.array(ground_points).shape)
        vertices=np.concatenate([np.array(vertices),np.array(ground_points)],axis=0)
        indices=np.concatenate([np.array(indices),np.array(ground_index)],axis=0)

        #Delaunay triangulation for ground points
        #saves respective indices in closing_indices
        tri = Delaunay(np.array(ground_points)[:,:2],furthest_site=False,incremental=False)
        closing_indices=tri.simplices
        
#         print(len(closing_indices))
        #converts received closing_indices (regarding closing_vertices) to the indices regarding vertices
        closing_indices += num_verts
        
        indices=np.concatenate([np.array(indices),np.array(closing_indices)],axis=0)
    
        vertices=np.array(vertices)
        
        #changes order of coordinates in vertices back
        #e.g. for y-method [x,z,y] --> [x,y,z]
        if transform==1 or transform == 2:
            axis=abs(transform-2)        #from [012] --> [210]
            vertices[:,[2,axis]] = vertices[:,[axis,2]]
            
        return vertices.tolist(),indices.tolist()
    
    def create_ground_points(self,point,minimum,maximum):
        new_grounds=[]
        corner=False
#         print('point',point)
        point=point.tolist()
#         image=(self.image_shape[0]-dx,self.image_shape[1]-dy)
#         print(image)
#         print(image[1])
#         xrange=np.arange(image[0]-dx/2, image[0]+dx/2, step=0.5)
#         yrange=np.arange(image[1]-dy/2, image[1]+dy/2, step=0.5)
#         assert point[1]==0 or (point[1] in yrange) or point[0]==0 or (point[0] in xrange)
        new_point=point.copy()
        if point[0]==0:
            new_point[0]+=1
            new_grounds.append(new_point)
            new_point=point.copy()
        if point[1]==0:
            new_point[1]+=1
            new_grounds.append(new_point)
            new_point=point.copy()
        if point[0] == maximum[0]:
            new_point[0]-=1
            new_grounds.append(new_point)
            new_point=point.copy()
        if point[1] == maximum[1]:
            new_point[1]-=1
            new_grounds.append(new_point)
            
        assert len(new_grounds)
        
        if len(new_grounds)!=1:
            corner=True
            new_grounds=0     
        
#         if point[0]==0 and point[1] in yrange:
#             new_grounds[0], new_grounds[1] = new_grounds[1], new_grounds[0]
        
#         print('new_grounds',new_grounds)

        return np.squeeze(new_grounds), corner
    
    def edge_points_start_at_corner(self,edge_points,minimum,maximum):
        new_points=edge_points.copy()
        k=0
        for index,point in enumerate(edge_points):
            p, corner = self.create_ground_points(point,minimum,maximum)
            if corner:
                break
            else:
                new_points.append(new_points[0])
                del new_points[0]
        return new_points
                                    
                                    
    def process(self,meshes,mesh_positions):
        intersect=[]
        difference=[]
        additional=[]
        for d in range(0,3):
            if len(meshes[d])!=0:
                if not mesh_positions is None:
                    assert len(mesh_positions[d])==len(meshes[d])
                    inter, diff, addit = self.cases(meshes[d],mesh_positions[d])
                    intersect.append(inter)
                    difference.append(diff)
                    additional.append(addit)
                else: print('mesh_position list is missing one entry')

        intersect = [x for x in intersect if x]
        difference = [x for x in difference if x]
        additional = [x for x in additional if x]
        
#         while len(difference):
#             print('in difference')
#             print(len(difference))
#             print(difference)
#             intersect[0][0] = self.difference(intersect[0][0],difference[0][0])
#             difference = np.delete(difference,0,0)
#             print(difference)
#             print(len(difference))
        
        main_inter=intersect[0][0]
        for diff in difference:
            print(diff)
            main_inter = self.difference(main_inter,diff[0])
            print('hi diff')
            
        for inter in range(1,len(intersect)):  
            main_inter = self.intersection(main_inter,intersect[inter][0])
            
        
#         while len(intersect)>1:
#             print('in intersect')
#             print(intersect)
#             print(intersect[0][0])
#             intersect[0][0] = self.intersection(intersect[0][0],intersect[1][0])
#             intersect = np.delete(intersect,1,0)
        
#         print(intersect[0][0].is_watertight)
        
        return main_inter, additional
                                      
    def cases(self,meshes,mesh_positions):
        intersect=[]
        difference=[]
        middle_intersect=[]
        middle_difference=[]
        additional=[]
        for mesh,position in zip(meshes,mesh_positions):
#             print('num verts pymesh',mesh.num_vertices)
            print('mesh vertices',mesh.vertices)
            if position == 'ou':
                intersect.append(mesh)
            elif position == 'ol':
                difference.append(mesh)
            elif position == 'iu':
                middle_intersect.append(mesh)
            elif position == 'il':
                middle_difference.append(mesh)
            else:
                print('Error in mesh position list.. Valid entries are: ou, ol, iu, il')
            
        if len(middle_intersect) == len(middle_difference) and len(middle_difference):
            for surf1, surf2 in zip(middle_intersect,middle_difference):
                difference.append(self.intersection(surf1,surf2))
        else:
            additional=np.concatenate([middle_intersect,middle_difference])
                
        return intersect, difference, additional
            
#     def union(self,surface_1,surface_2):
        
        
        
    def difference(self,mesh_1,mesh_2):
        print(mesh_1.is_watertight)
        print(mesh_2.is_watertight)
        if not (mesh_1.is_watertight and mesh_2.is_watertight):
            print('Strange results expected after boolean operation: difference, since at least one of the two meshes is not watertight')
#             adj, adj_verts = graph.face_adjacency(mesh=mesh_1,return_edges=True)
#             adj2, adj_verts_2 = graph.face_adjacency(mesh=mesh_2,return_edges=True)
#             fac = mesh_1.faces
#             edg = mesh_1.edges
#             edg2 = mesh_2.edges
# #             edg2 = edg
#             print('adj vert shape',np.array(adj_verts).shape)
#             print('edges shape',np.array(edg).shape)
#             for a in adj_verts:
#                 k=0
#                 a_tilde=[0,0]
#                 a_tilde[0],a_tilde[1] =a[1],a[0]
# #                 print(a)
#                 for i,edge in enumerate(edg):
#                     if np.array_equal(edge,a) or np.array_equal(edge,a_tilde):
#                         edg=np.delete(edg,k,0)
# #                         print('hi',k)
#                         k-=1
#                     k+=1
#             print(edg.shape)                        
#             print(edg)  
#             print(mesh_1.vertices.shape)
#             print('verts',mesh_1.vertices[edg])
#             print('adj_verts shape',adj_verts_2.shape)
#             print('edges shape',edg2.shape)
#             for a in adj_verts_2:
#                     k=0
#                     a_tilde=[0,0]
#                     a_tilde[0],a_tilde[1] =a[1],a[0]
#     #                 print(a)
#                     for i,edge in enumerate(edg2):
#                         if np.array_equal(edge,a) or np.array_equal(edge,a_tilde):
#                             edg=np.delete(edg2,k,0)
#     #                         print('hi',k)
#                             k-=1
#                         k+=1
#             print(edg2.shape)
#             print('verts',mesh_2.vertices[edg2])
        mesh = [mesh_1, mesh_2]
#         mesh = pymesh.boolean(mesh_1, mesh_2, "difference",engine="carve")
        mesh = boolean.difference(mesh,'blender')
        print(mesh.is_watertight)
#         print('num verts pymesh in difference',mesh.num_vertices)
        return mesh
        
    def intersection(self,mesh_1,mesh_2):
        print(mesh_1.is_watertight)
        print(mesh_2.is_watertight)
        if not (mesh_1.is_watertight and mesh_2.is_watertight):
            print('Strange results expected after boolean operation: intersection, since at least one of the two meshes is not watertight')
        mesh = [mesh_1, mesh_2]
        mesh = boolean.intersection(mesh,'blender')
#         mesh = pymesh.boolean(mesh_1, mesh_2, "intersection")
#         print('num verts pymesh in intersection',mesh.num_vertices)
        return mesh

    def get_verts(self,mesh):
        print(mesh.is_watertight)
#         adj, adj_verts = graph.face_adjacency(mesh=mesh,return_edges=True)
#         fac = mesh.faces
#         edg = mesh.edges
#         print('adj vert shape',np.array(adj_verts).shape)
#         print('edges shape',np.array(edg).shape)
#         for a in adj_verts:
#             k=0
#             a_tilde=[0,0]
#             a_tilde[0],a_tilde[1] =a[1],a[0]
# #                 print(a)
#             for i,edge in enumerate(edg):
#                 if np.array_equal(edge,a) or np.array_equal(edge,a_tilde):
#                     edg=np.delete(edg,k,0)
# #                         print('hi',k)
#                     k-=1
#                 k+=1
#         print(edg.shape)                        
#         print(edg)  
#         print(mesh.vertices.shape)
#         print('verts',mesh_1.vertices[edg])
        verts = mesh.vertices
        inds = mesh.faces
        normals = mesh.vertex_normals
#         mesh.add_attribute("vertex_normal")
#         normals = mesh.get_attribute("vertex_normal")
#         print(normals.shape)
#         num_normals = int(normals.shape[0])
#         normals=normals.reshape(int(num_normals/3),3)
#         normals=None
        
        return verts, inds, normals
    
    def get_verts2(self,mesh):
        verts = mesh.vertices
        inds = mesh.faces
        mesh.enable_connectivity()
        adjacent_faces={}
        for f in range(len(inds)):            
            adjacent_faces[f]=mesh.get_face_adjacent_faces(f)
#         print(adjacent_faces)
        critical=[]
        for af in adjacent_faces:
            print(len(adjacent_faces[af]))
            if len(adjacent_faces[af])!=3:
                critical.append(np.append([af],adjacent_faces[af],axis=0))
        print(critical)
        mesh, info = pymesh.remove_isolated_vertices(mesh)
        mesh,info = pymesh.remove_duplicated_vertices(mesh)
        mesh,info = pymesh.remove_duplicated_faces(mesh)
        return mesh
    
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
        