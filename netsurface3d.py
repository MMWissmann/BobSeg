import numpy as np
import bresenham as bham
import maxflow
import math

from spimagine import EllipsoidMesh, Mesh

class NetSurf3d:
    """
    Implements the optimal net surface problem for multiple surfaces.
    Relevant publication: [Li 2006]
    """
    
    INF = 9999999999
    
    image = None
    min_col_height = None
    max_col_height = None
    
    num_columns = None
    
    w = None
    w_tilde = None
    
    nodes = None
    edges = None
    g = None
    maxval = None
    
    def __init__(self, K=50, max_delta_k=4, dx=1, dy=1, surfaces=1, min_dist=0, max_dist=20):
        """
        Parameters:
            K           -  how many sample points per column
            max_delta_k -  maximum column height change between neighbors (as defined by adjacency)
            dx/dy       -  divisor: choose only every d-th pixel to create a column on base graph, for x and y
            surfaces    -  number of surfaces to be detected
            min/max_dist-  for more than 1 surface: minimum and maximum intersurface distances !in terms of K!
        """           
        self.dx = dx
        self.dy = dy
        self.K = K
        self.max_delta_k = max_delta_k
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.surfaces = surfaces
        
    def base_points(self, dx, dy, mask=None):
        """
        choose every dxth point in x-direction (dyth point in y-direction) to be vertex on base graph
        returns (x,y) coordinates of these vertices
           self.num_columns: total number of columns/ total number of vertices in base graph
        """ 
        if not mask is None: assert(self.image.shape[1,2] == mask.shape)
        
        image_length_x = self.image.shape[1]
        image_length_y = self.image.shape[2]
        
        points = []
        
        if dx > image_length_x or dy > image_length_y:
            print('Number of columns smaller than 1 in at least one direction.. Choose smaller dx or dy')
            return None
        else:
            for i in range(int(image_length_x/dx)):
                for j in range(int(image_length_y/dy)):
                    if (i*dx) <= image_length_x and (j*dy) <= image_length_y:
                        (x,y) = (int(i * dx),int(j * dy))
                        if mask is None:
                            points.append((x,y))
                        else:
                            if self.mask_pixel_value(mask,(x,y)) is True:
                                points.append((x,y))
                            else: continue
                    else: continue
                        
        self.num_columns=len(points)
        
        self.num_base_y=int(image_length_y/dy)
        
        return np.array(points)
                                
    def mask_pixel_value(self,mask,pixel):
        try:
            m = mask[pixel]
        except:
            None
        
        assert m==1 or m==0
        
        if m == 1: return True
        elif m == 0: return False
    
    def apply_to(self, image, max_col_height, min_col_height=0, mask=None):  
        '''
        image: input 3d image
        max_col_height / min_col_height: maximal and minimal z coordinate at which a surface is expected
        mask: optional input of 2d image to be used to select coordinates for base graph generation with:   
                1. same (x,y) dimensions as original 3d image,
                2. only 1 or 0 as pixel values
        '''
        assert( len(image.shape) == 3 )
        if max_col_height > image.shape[0]:
            max_col_height = image.shape[0]
            print('maximal column height exceeds image boundary, set to', max_col_height)
            print('object expected between %d and' %(min_col_height), max_col_height)
        
        self.image = image
        self.min_col_height = min_col_height
        self.max_col_height = max_col_height
        
        self.base_coords = self.base_points(self.dx, self.dy, mask)
                                        
        print('Number of columns:', self.num_columns)
        
        self.compute_weights()
        self.build_flow_network()
        
        self.maxval = self.g.maxflow()
        return self.maxval
    
    def compute_weights(self):        
        assert not self.image is None
        
        self.w = np.zeros([self.num_columns, self.K]) # node weights
        self.w_tilde = np.zeros([self.num_columns, self.K])
        
        for i in range(self.num_columns):
            from_xyz = np.array([[int(self.base_coords[i,0]),int(self.base_coords[i,1]),int(self.min_col_height)]])
            to_xyz   = np.array([[int(self.base_coords[i,0]),int(self.base_coords[i,1]),int(self.max_col_height)]])
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
        print('Number of Surfaces:', self.surfaces)
        self.num_nodes = self.surfaces*self.num_columns*self.K
        # estimated num edges (in case I'd have 4 num neighbors and full pencils)
        self.num_edges = ( self.num_nodes * 4 * (self.max_delta_k + self.max_delta_k+1) ) * .5

        self.g = maxflow.Graph[float]( self.num_nodes)
        self.nodes = self.g.add_nodes( self.num_nodes )
        
        NoneType = type(None)
        
        for s in range(self.surfaces):
            
            c=s*self.num_columns*self.K #total number of nodes already added with the surfaces above
            c_above=(s-1)*self.num_columns*self.K #not relevant for surface 1 (s=0)
            print('c',c)

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
                                self.g.add_edge(c+i*self.K+k, c+j*self.K+k2, alpha, 0)    
                        else: continue
                
                # create intersurface connections, if more than one surface
                if 0 < s:
                    if i==0:
                        #making sure that base set is strongly connected
                        self.g.add_edge(c_above, c, self.INF,0)
                    for k in range(self.K):
                        #introducing max intersurface distance
                        k2 = max(0,k-self.max_dist)
                        self.g.add_edge(c_above+i*self.K+k, c+i*self.K+k2, self.INF, 0)
                        #introducing min intersurface distance
                        k3 = min(self.K,k+self.min_dist)
                        self.g.add_edge(c+i*self.K+k, c_above+i*self.K+k3, self.INF, 0)
                            
    def calculate_neighbors_of(self,point):
        '''
        Determine the column-id's of the 2,3, or 4 neighboring columns of point
        Will be called by build_flow_network()
        '''
        neighbors_of = np.zeros([4])
        y=self.num_base_y
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
        myverts = np.zeros((self.surfaces*self.num_columns,3))
        for s in range(self.surfaces):
            for i in range(s*self.num_columns, self.num_columns+s*self.num_columns):
                    myverts[i] = self.get_surface_point(i,s)
        return myverts        
            
    def get_surface_point( self, column_id, s ):
        '''
        For column_id in g, the last vertex that is still in S (of the s-t-cut) is determined.
        Note, that column_id does not represent the respective array of voxels (if nx,ny != 1,1)
        Note, that column_id does also not nessecarily represent the respective column of the base graph (if surfaces>1)
        '''
        for k in range(self.K):
            if self.g.get_segment(column_id*self.K+k) == 1: break # leave as soon as k is first outside point
        k-=1
        x = int(self.base_coords[column_id-s*self.num_columns,0])
        y = int(self.base_coords[column_id-s*self.num_columns,1])
        z = int(self.min_col_height + (k-1)/float(self.K) * (self.max_col_height-self.min_col_height))
        return (x,y,z)
    
    '''
    VISUALISATION STUFF
    '''

    def get_triangles(self):
        '''
        returns array of indices of three columns for triangulation of surface mesh
        '''
        y=self.num_base_y
        tris=[]
        
        for i in range(self.num_columns-y):
            if (i+1) % y == 0:
                continue
            else:
                tris.append((int(i),int(i+1),int(i+y)))
        
        for i in range(y,self.num_columns):
            if (i) % y == 0:
                continue
            else:
                tris.append((int(i),int(i-y),int(i-1)))
                
        tris=np.array(tris)
        tris=tris.astype(np.int)
        
        return tris
    
    def norm_vec(self,arr):
        ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
        lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
        arr[:,0] /= lens
        arr[:,1] /= lens
        arr[:,2] /= lens                
        return arr
   
    def norm_radii(self,cabs,pixelsizes):
        """ 
        converts from absolute pixel based radii to normalized [0,1] coordinates for spimagine meshes (z,y,x).
        """        
        cnorm = 2. * np.array(cabs[::-1], float) / np.array(pixelsizes)
        return tuple(cnorm[::-1])
    
    def get_normals(self,faces,verts):
        '''
        get normal vectors for spimagine mesh generation
        '''
        norm = np.zeros( verts.shape, dtype=verts.dtype )
        tris = verts[faces]      
        n = -np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
        n = abs(n)
        n=self.norm_vec(arr=n)
        norm[ faces[:,0] ] += n
        norm[ faces[:,1] ] += n
        norm[ faces[:,2] ] += n
        norm=self.norm_vec(arr=norm)
        return norm[faces]
        
    def norm_coords(self,cabs,pixelsizes):
        """ 
        converts from absolute pixel location in image (x,y,z) to normalized [0,1] coordinates for spimagine meshes (z,y,x).
        """        
        cnorm = 2. * np.array(cabs, float) / np.array(pixelsizes) -1.
        return tuple(cnorm)
    
    def create_surface_mesh( self, s, facecolor=(1.,.3,.2) ):
        '''
        Generates one spimagine Mesh of the surface s
        -surface vertices determined by get_surface_point
        -indices that choose which vertices to triangulate given by get_triangles
        '''
        
        image_shape_xyz = (self.image.shape[1],self.image.shape[2], self.image.shape[0])
        myverts = np.zeros((self.num_columns, 3))        
        x=np.zeros((self.num_columns,3))
        
        myindices=self.get_triangles()
        verts=np.zeros((myindices.shape[0]*3,3))
        
        j=0  
        
        for i in range(s*self.num_columns, self.num_columns+s*self.num_columns):
            p = self.get_surface_point(i,s)
            myverts[j,:] = self.norm_coords( p, image_shape_xyz )
            x[j]=p
            j+=1
            
        
        k=0
        for l in myindices:
            for m in l:
                verts[k]=x[m]
                verts[k]=self.norm_coords(verts[k],image_shape_xyz)
                k+=1
        
        assert myverts.shape[0] == self.num_columns
        #mynormals=self.get_normals(myindices,myverts)
        #myverts=np.concatenate((myverts,myverts),axis=0)
        print('myindices', myindices.shape)
        #print(mynormals.shape)
        ind=np.arange(0,3*myindices.shape[0])
        print('ind', ind.shape)
        print('myverts', myverts.shape)
        print('verts', verts.shape)
        mynormals=self.get_normals(myindices,myverts)
        print('normals', mynormals.shape)
        mynormals=self.norm_radii(mynormals,image_shape_xyz)
        indices=ind.tolist()
        print(len(indices))

        return Mesh(vertices=verts, indices=indices, normals=mynormals, facecolor=facecolor, alpha=.5)