import numpy as np
import bresenham as bham
import maxflow
import math
from scipy.spatial import Delaunay

from spimagine import EllipsoidMesh, Mesh

class NetSurfBottom:
    """
    Implements the optimal net surface problem for multiple surfaces.
    Relevant publication: [Li 2006]
    """
    
    INF = 9999999999
    
    image = None
    min_dist_1 = None
    max_dist_1 = None
    
    num_columns = None
    
    w = None
    w_tilde = None
    
    nodes = None
    edges = None
    g = None
    maxval = None
    
    def __init__(self, K=50, max_delta_k=4, divx=1, divy=1, min_dist=0, max_dist=50, s=1):
        """
        Parameters:
            K           -  how many sample points per column
            max_delta_k -  maximum column height change between neighbors (as defined by adjacency)
            div         -  divisor: choose only every divth pixel to create a column on base graph, for x and y
            dist        -  minimum and maximum intersurface distances in terms of K
            s           -  number of surfaces to be detected
        """           
        self.divx = divx
        self.divy = divy
        self.K = K
        self.max_delta_k = max_delta_k
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.surfaces = s
        
    def base_points(self, nx, ny):
        """
        choose every nxth point in x-direction (nyth point in y-direction) to be vertex on base graph
        returns (x,y) coordinates of these vertices
        self.num_columns: total number of columns
        self.ymax/xmax: total number of columns in 1D (y and x)
        """
        x_frame = self.image.shape[1]
        y_frame = self.image.shape[2]
        
        points = np.zeros([int((x_frame/nx)*y_frame/ny), 2])
        x=0
        y=0
        
        ycol=0
        xcol=0
        
        if nx > x_frame or ny > y_frame:
            print('Number of columns smaller than 1 in at least one direction.. Choose smaller divisor')
            return None
        else:
            for i in range(int(x_frame/nx)):
                for j in range(int(y_frame/ny)):
                    if (nx+i*nx) <= x_frame and (ny+j*ny) <= y_frame:
                        x = int(nx + i * nx)
                        y = int(ny + j * ny)
                        points[j+i*int(y_frame/ny)] = [x,y]
                    else: continue
                    ycol=max(ycol,j)
                xcol=max(xcol,i)
        
        xcol+=1
        ycol+=1
        
        nonzero_row_indices =[i for i in range(points.shape[0]) if not np.allclose(points[i,:],0)]
        points = points[nonzero_row_indices,:] #taking remaining zeros out of points
             
        self.xmax=xcol
        self.ymax=ycol
        
        self.num_columns=points.shape[0]
        
        assert self.num_columns == self.ymax * self.xmax
       
        return points
    
    def apply_to(self, image, max_dist_1, min_dist_1=0):        
        assert( len(image.shape) == 3 )
        
        self.image = image
        self.min_dist_1 = min_dist_1
        self.max_dist_1 = max_dist_1
        
        self.col_vectors = self.base_points(self.divx, self.divy)
                                        
        print(self.num_columns)
        
        self.compute_weights()
        self.build_flow_network()
        
        self.maxval = self.g.maxflow()
        return self.maxval
    
    def compute_weights(self):        
        assert not self.image is None
        
        self.w = np.zeros([self.num_columns, self.K]) # node weights
        self.w_tilde = np.zeros([self.num_columns, self.K])
        
        for i in range(self.num_columns):
            from_xyz = np.array([[int(self.col_vectors[i,0]),int(self.col_vectors[i,1]),int(self.min_dist_1)]])
            to_xyz   = np.array([[int(self.col_vectors[i,0]),int(self.col_vectors[i,1]),int(self.max_dist_1)]])
            coords = bham.bresenhamline(from_xyz, to_xyz)
            num_pixels = len(coords)
            for k in range(self.K):
                start = int(k * float(num_pixels)/self.K)
                end = max( start+1, start + int(num_pixels/self.K) )
                self.w[i,k] = -1 * self.compute_weight_at(coords[start:end])
            
        for i in range(self.num_columns):
            self.w_tilde[i,0] = self.w[i,0] 
            for k in range(1,self.K):
                self.w_tilde[i,k] = self.w[i,k]-self.w[i,k-1]

    def compute_weight_at( self, coords ):
        '''
        coords  list of lists containing as many entries as img has dimensions
        '''
        m = 0
        for c in coords:
            try:
                m = max( m,self.image[ tuple(c[::-1]) ] )
            except:
                None
        return m
    
    def build_flow_network( self, alpha=None):
        '''
        Builds the flow network that can solve the V-Weight Net Surface Problem
        Returns a tuple (g, nodes) consisting of the flow network g, and its nodes.
        
        If alpha != None this method will add an additional weighted flow edge (horizontal binary costs.
        '''
        self.num_nodes = self.surfaces*self.num_columns*self.K
        # estimated num edges (in case I'd have 4 num neighbors and full pencils)
        self.num_edges = ( self.num_nodes * 4 * (self.max_delta_k + self.max_delta_k+1) ) * .5

        self.g = maxflow.Graph[float]( self.num_nodes, self.num_edges)
        self.nodes = self.g.add_nodes( self.num_nodes )
        
        NoneType = type(None)
        
        for s in range(self.surfaces):
            
            c=s*self.num_columns
            c_above=(s-1)*self.num_columns

            for i in range( self.num_columns ):

                # connect column to s,t
                for k in range( self.K ):
                    if self.w_tilde[i,k] < 0:
                        self.g.add_tedge(c+i*self.K+k, -self.w_tilde[i,k], 0)
                    else:
                        self.g.add_tedge(c+i*self.K+k, 0, self.w_tilde[i,k])

                # connect column to i-chain
                for k in range(1,self.K):
                    self.g.add_edge(c+i*self.K+k, c+i*self.K+k-1, self.INF, 0)

                # connect column to neighbors
                for k in range(self.K):
                    for j in self.calculate_neighbors_of(i):
                        if not isinstance(j,NoneType):
                            k2 = max(0,k-self.max_delta_k)
                            self.g.add_edge(c+i*self.K+k, c+j*self.K+k2, self.INF, 0)
                            if alpha != None:
                                # add constant cost penalty \alpha
                                self.g.add_edge(c+i*self.K+k, c+j*self.K+k, alpha, 0)    
                        else: continue
                
                # create intersurface connections, if more than one surface
                if 0 < s <= self.surfaces-1:
                    if i==0:
                        #making sure that base set is strongly connected
                        self.g.add_edge(c_above, c, self.INF,0)
                    for k in range(self.K):
                        if k >= self.max_dist:
                            #introducing max intersurface distance
                            self.g.add_edge(c_above+i*self.K+k, c+i*self.K+k-self.max_dist, self.INF, 0)
                        elif k < self.K - self.min_dist:
                            #introducing min intersurface distance
                            self.g.add_edge(c+i*self.K+k, c_above+i*self.K+k+self.min_dist, self.INF, 0)
                        else: continue
                            
    def calculate_neighbors_of(self,point):
        '''
        Determine the column-id's of the 2,3, or 4 neighboring columns of point
        Will be called by build_flow_network()
        '''
        neighbors_of = np.zeros([4])
        y=self.ymax
        a = point +1
        b = point -1
        c = point +y
        d = point -y
        if not ( ( point % y == 0 ) or ( (point +1) % y == 0 ) ):
            if not ( point < y or point > (self.num_columns - y) ):
                neighbors_of=[a,b,c,d]
            elif point < y:
                neighbors_of=[a,b,c, None]
            elif point > (self.num_columns - y):
                neighbors_of=[a,b,d, None]
            else:
                print("sth went wrong in determining neighbors of", point)
        elif point == 0:
            neighbors_of=[a,c, None, None]
        elif point % y == 0 and not point == 0:
            if point == self.num_columns - y:
                neighbors_of=[a,d, None, None]
            else:
                neighbors_of=[a,c,d, None]
        elif (point +1) % y == 0:
            if point + 1 == y:
                neighbors_of=[b,c, None, None]
            elif point +1 == self.num_columns:
                neighbors_of=[b,d, None, None]
            else:
                neighbors_of=[b,c,d, None]
        else: print("error in calculate_neighbors_of", point)
        return neighbors_of

    def get_counts( self ):
        size_s_comp = 0
        size_t_comp = 0
        for n in self.nodes:
            seg = self.g.get_segment(n)
            if seg == 0:
                size_s_comp += 1
            else:
                size_t_comp += 1
        return size_s_comp, size_t_comp
    
    
    def give_surface_points( self):
        myverts = np.zeros((self.num_columns,3))
        for i in range(self.num_columns):
            myverts[i] = self.get_surface_point(i)
        return myverts
            
    def get_volume( self, calibration = (1.,1.,1.) ):
        """
        calibration: 3-tupel of pixel size multipliers
        """
        volume = 0.
        for a,b,c in self.triangles:
            pa = self.get_surface_point( a )
            pb = self.get_surface_point( b )
            pc = self.get_surface_point( c )    
            volume += self.get_triangle_splinter_volume( pa, pb, pc, calibration )

        return volume   
         
            
    def get_surface_point( self, column_id ):
        for k in range(self.K):
            if self.g.get_segment(column_id*self.K+k) == 1: break # leave as soon as k is first outside point
        k-=1
        x = int(self.col_vectors[column_id,0])
        y = int(self.col_vectors[column_id,1])
        z = int(self.min_dist_1 + (k-1)/float(self.K) * (self.max_dist_1-self.min_dist_1))
        return (x,y,z)
    
    def get_triangle_splinter_volume( self, pa, pb, pc, calibration ):
        """
        Computes the volume of the pyramid defined by points pa, pb, pc, and self.center
        """
        assert not self.center is None
        
        x = (np.array(pa)-self.center) * calibration[0]
        y = (np.array(pb)-self.center) * calibration[1]
        z = (np.array(pc)-self.center) * calibration[2]
        return math.fabs( x[0] * y[1] * z[2] + 
                          x[1] * y[2] * z[0] + 
                          x[2] * y[0] * z[1] - 
                          x[0] * y[2] * z[1] - 
                          x[1] * y[0] * z[2] - 
                          x[2] * y[1] * z[0]) / 6.


    '''
    VISUALISATION STUFF
    '''

    def get_triangles(self):
        '''
        returns array of indices of three columns for triangulation of surface mesh
        '''
        y=self.ymax
        x=2*self.num_columns
        tris=np.zeros([x,3])
        for i in range(self.num_columns-y):
            if (i+1) % y == 0:
                continue
            else:
                tris[i]=[int(i),int(i+1),int(i+y)]
        
        for i in range(y,self.num_columns):
            if (i) % y == 0:
                continue
            else:
                tris[self.num_columns+i]=[int(i),int(i-1),int(i-y)]

        nonzero_row_indices =[i for i in range(tris.shape[0]) if not np.allclose(tris[i,:],0)]
        tris = tris[nonzero_row_indices,:] #taking remaining zeros out of tris
        tris=tris.astype(int)
        print(tris)
        
        return tris
    
    def norm_vec(self,arr):
        ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
        lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
        arr[:,0] /= lens
        arr[:,1] /= lens
        arr[:,2] /= lens                
        return arr
    
    def get_normals(self,faces,verts):
        '''
        get normal vectors for spimagine mesh generation
        '''
        norm = np.zeros( verts.shape, dtype=verts.dtype )
        verts=verts
        tris = verts[faces]      
        n = -np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
        n=self.norm_vec(arr=n)
        print('in get normals')
        print(faces.shape)
        print(verts.shape)
        print(tris.shape)
        print(n.shape)
        norm[ faces[:,0] ] += n
        norm[ faces[:,1] ] += n
        norm[ faces[:,2] ] += n
        norm=self.norm_vec(arr=norm)
        print(faces[:,0])
        return norm[faces]
        
    def norm_coords(self,cabs,pixelsizes):
        """ 
        converts from absolute pixel location in image (x,y,z) to normalized [0,1] coordinates for spimagine meshes (z,y,x).
        """        
        cnorm = 2. * np.array(cabs[::-1], float) / np.array(pixelsizes) - 1.
        return tuple(cnorm[::-1])

    def norm_radii(self,cabs,pixelsizes):
        """ 
        converts from absolute pixel based radii to normalized [0,1] coordinates for spimagine meshes (z,y,x).
        """        
        cnorm = 2. * np.array(cabs[::-1], float) / np.array(pixelsizes)
        return tuple(cnorm[::-1])
    
    
    def create_surface_mesh( self, facecolor=(1.,.3,.2) ):
        myverts = np.zeros((self.num_columns, 3))
        x=np.zeros((self.num_columns,3))
        myindices=self.get_triangles()
        
        for i in range(self.num_columns):
            p = self.get_surface_point(i)
            myverts[i,:] = self.norm_coords( p, self.image.shape )
            x[i]=p
        mynormals=self.get_normals(myindices,x)
        myverts=np.concatenate((myverts,myverts),axis=0)
        print(myverts.shape)
        print(myindices.shape)
        print(mynormals.shape)

        return Mesh(vertices=myverts, normals = mynormals, indices=myindices,  facecolor=facecolor, alpha=.5)