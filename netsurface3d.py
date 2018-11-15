import numpy as np
import bresenham as bham
import maxflow
import math

from scipy.spatial import Delaunay
from spimagine import EllipsoidMesh, Mesh
import matplotlib.pyplot as plt

from stitch_surfaces import StitchSurfaces
from CrossSection import CrossSection

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
    
    def __init__(self, K=50, max_delta_k=4, dx=1, dy=1, surfaces=1, min_dist=0, max_dist=50,axis=0):
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
        self.axis = axis
    
    def apply_to(self, image, max_col_height, min_col_height=0, mask=None, plot_base_graph=False, solveAsIlp=False):  
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
        self.neighbors = self.neighbors(mask,plot_base_graph)
                                        
        print('Number of columns:', self.num_columns)
        
        self.compute_weights()
        self.build_flow_network()
        
#        print('Only thing happening from now on: PyMaxFlow calculates.')
        self.maxval = self.g.maxflow()
        return self.maxval
#         return None
    
    def base_points(self, dx, dy, mask=None):
        """
        choose every dxth point in x-direction (dyth point in y-direction) to be vertex on base graph
        returns (x,y) coordinates of these vertices
        self.num_columns: total number of columns/ total number of vertices in base graph
        """ 
        if not mask is None: 
            assert(self.image[0].shape == mask.shape)
        
        image_length_x = self.image.shape[2]
        image_length_y = self.image.shape[1]
        
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
        self.num_base_x=int(image_length_x/dx)
        
        return np.array(points)
                                
    def mask_pixel_value(self,mask,pixel):
        
        try:
            m = mask[pixel[::-1]]
            if m == 1: return True
            elif m == 0: return False
        except:
            None
        
    def neighbors(self,mask,plot_base_graph=False):
        '''
        Perform Delaunay triangulation with base graph coordinates in order to find neighbored columns
        Returns adjacency list --> Array of neighboring vertices of vertex k is: self.neighbors[k] 
        '''
        tri = Delaunay(self.base_coords)
        
        if plot_base_graph is not False:
            plt.triplot(self.base_coords[:,0], self.base_coords[:,1], tri.simplices.copy())
            plt.plot(self.base_coords[:,0], self.base_coords[:,1], 'o')
            plt.show()
        
        triangles = tri.simplices
        
        #produce adjacency list
        neighbors = {}
        
        for point in range(self.num_columns):
            neighbors[point] = []

        for simplex in tri.simplices:
            neighbors[simplex[0]] += [simplex[1],simplex[2]]
            neighbors[simplex[1]] += [simplex[2],simplex[0]]
            neighbors[simplex[2]] += [simplex[0],simplex[1]]
        
        #remove doubles and sort by value
        for a in range(len(neighbors)):
            neighbors[a]=list(set(neighbors[a]))
            neighbors[a]=np.sort(neighbors[a])
        
        remove_triangles=[]
        
        #remove edges that connect through 0-part of mask (allowing non-convex shapes of the base graph)
        if mask is None:
            self.triangles=triangles
        else:
            for a in range(len(neighbors)):
                k=-1
                for n in neighbors[a]:
                    k+=1
                    from_xy = np.array([[int(self.base_coords[a,0]),int(self.base_coords[a,1])]])
                    to_xy = np.array([[int(self.base_coords[n,0]),int(self.base_coords[n,1])]])
                    line = bham.bresenhamline(from_xy,to_xy)
                    num_pixels = len(line)
                    for coord in line:
                        pix = self.mask_pixel_value(mask,tuple(coord))
                        if pix is False:
                            neighbors[a]=np.delete(neighbors[a],k)
                            remove_triangles.append((a,n))
                            k-=1
                            break
            remove_triangles=np.array(remove_triangles)
        
            #remove triangles from tri.simplices where at least one edge connects through 0-part of mask for visualisation
            self.triangles=self.spim_tris(triangles,remove_triangles)
                    
        return neighbors
    
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
                
        #self.w=self.w[:,::-1]
            
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
            except: None
        return m
    
    def build_flow_network( self, alpha=None):
        '''
        Builds the flow network that can solve the V-Weight Net Surface Problem
        Returns a tuple (g, nodes) consisting of the flow network g, and its nodes.
        
        If alpha != None this method will add an additional weighted flow edge (horizontal binary costs.
        '''
        
        # To measure timing
        #' http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
        def tic():
            # Homemade version of matlab tic and toc functions
            import time
            global startTime_for_tictoc
            startTime_for_tictoc = time.time()

        def toc():
            import time
            if 'startTime_for_tictoc' in globals():
                print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
            else:
                print("Toc: start time not set")
        
        
        print('Number of Surfaces:', self.surfaces)
        self.num_nodes = self.surfaces*self.num_columns*self.K
        # estimated num edges (in case I'd have 4 num neighbors and full pencils)
        #self.num_edges = ( self.num_nodes * 4 * (self.max_delta_k + self.max_delta_k+1) ) * .5

        self.g = maxflow.Graph[float]( self.num_nodes)
        self.nodes = self.g.add_nodes( self.num_nodes )
        
        for s in range(self.surfaces):
            
            c=s*self.num_columns*self.K #total number of nodes already added with the surfaces above
            c_above=(s-1)*self.num_columns*self.K #not relevant for surface 1 (s=0)
#             print('c',c)
            tic()

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
                    for j in self.neighbors[i]:
                        k2 = max(0,k-self.max_delta_k)
                        self.g.add_edge(c+i*self.K+k, c+j*self.K+k2, self.INF, 0)
                        if alpha != None:
                            # add constant cost penalty \alpha
                            self.g.add_edge(c+i*self.K+k, c+j*self.K+k2, alpha, 0)    
                
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
            print('Done with surface',s)
            toc()            
                                       
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
        myverts = {}
        for s in range(self.surfaces):
            for i in range(s*self.num_columns, self.num_columns+s*self.num_columns):
                    myverts[s]=self.get_surface_point(i,s)
        return np.array(myverts)        
            
    def get_surface_point( self, column_id, s ):
        '''
        For column_id in g, the last vertex that is still in S (of the s-t-cut) is determined.
        Note, that column_id does not represent the respective array of voxels (if dx,dy != 1,1)
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
    
    def spim_tris(self,triangles,remove_tris):
        '''delete triangles from delaunay-produced array that cover areas outside of mask'''
        k=-1
        for t in range(len(triangles)):
            k+=1
            for r in remove_tris:
                if any(np.isin(triangles[k],r[0])) and any(np.isin(triangles[k],r[1])):
                    triangles=np.delete(triangles,k,0)
                    k-=1
        return triangles
    
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
        tris = verts[faces]     # gives [[triangle1:[x,y,z],[x,y,z],[x,y,z]],[triangle2:[],[],[]]...]
        n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] ) #cross product every triangle (len(n)=len(faces))
        n=self.norm_vec(arr=n)     
        norm[ faces[:,0] ] += n
        norm[ faces[:,1] ] += n
        norm[ faces[:,2] ] += n
        norm=self.norm_vec(arr=norm)
        return norm
        
    def norm_coords(self,cabs,pixelsizes):
        """ 
        converts from absolute pixel location in image (x,y,z) to normalized [0,1] coordinates for spimagine meshes (z,y,x).
        performs translation operation for the mesh, so that every vertex is in the middle of the respective voxel
        """        
        cnorm = 2. * np.array(cabs, float) / np.array(pixelsizes) -1.  
        
        num_grid_z=min(self.K,self.image.shape[0])
        cnorm[0]+=1/self.num_base_x
        cnorm[1]+=1/self.num_base_y
        cnorm[2]+=3/num_grid_z
        
        return tuple(cnorm)

    def create_surface_mesh( self, s, facecolor=(1.,.3,.2), export=False):
        '''
        Generates one spimagine Mesh of the surface s
        -surface vertices determined by get_surface_point()
        -indices that choose which vertices to triangulate given by self.triangles in neighbors()
        '''
        
        image_shape_xyz = [self.image.shape[2],self.image.shape[1], self.image.shape[0]]
        
        xyz=np.zeros((self.num_columns,3))
        
        myindices=self.triangles
        verts=np.zeros((myindices.shape[0]*3,3))
        
        j=0  
        for i in range(s*self.num_columns, self.num_columns+s*self.num_columns):
            xyz[j]=self.get_surface_point(i,s)
            j+=1
        
        #rotate image_shape and vertice-coordinates back to original xyz in case of surface detection other than z-direction
        if self.axis!=0:
            axis=abs(self.axis-2)        #from [012] --> [210]
            image_shape_xyz[2], image_shape_xyz[axis] = image_shape_xyz[axis], image_shape_xyz[2]
            xyz[:,[2,axis]] = xyz[:,[axis,2]]
        
        if s == 0:
            self.vertices=xyz
        else:
            self.vertices=np.append(self.vertices,[xyz])
            self.vertices=np.reshape(self.vertices,(s+1,self.num_columns,3))
        
        k=0
        for l in myindices:
            for m in l:
                verts[k]=xyz[m]
                verts[k]=self.norm_coords(verts[k],tuple(image_shape_xyz))
                k+=1
        
        ind=np.arange(0,3*myindices.shape[0])
        self.mynormals=self.get_normals(myindices,xyz)
        indices=ind.tolist()
        
        if not export is False:
            self.save_surface(s,xyz,myindices,self.mynormals)
        
        mynormals = self.mynormals[myindices]

        return Mesh(vertices=verts, indices=indices, normals=mynormals, facecolor=facecolor, alpha=.5)
    
    def create_stitching_mesh( self, s, facecolor=(1.,.3,.2), export=False):
        '''
        Generates one spimagine stitching Mesh that connects surface s and surface s+1
        -surface vertices and indices determined by stitch() in StitchSurfaces Class
        '''
        image_shape_xyz = (self.image.shape[2],self.image.shape[1], self.image.shape[0]) 
        
        stitchsurfaces=StitchSurfaces(self.vertices,self.triangles) 
        stitch_surfaces=np.array([s,s+1])
        stitch_verts, stitch_indices = stitchsurfaces.stitch(stitch_surfaces)
        
        verts=np.zeros((stitch_indices.shape[0]*3,3))
        
        k=0
        for l in stitch_indices:
            for m in l:
                verts[k]=stitch_verts[m]
                verts[k]=self.norm_coords(verts[k],image_shape_xyz)
                k+=1
        
        ind=np.arange(0,3*stitch_indices.shape[0])
        mynormals=self.get_normals(stitch_indices,stitch_verts)
        indices=ind.tolist()
        
        if export is not False:
            self.save_surface(s,stitch_verts,stitch_indices,mynormals,'stitch')

        return Mesh(vertices=verts, indices=indices, normals=mynormals, facecolor=facecolor, alpha=.5)
    
    def show_sections(self,plane_orig,plane_normal,num_slices):
        
        cross_section=CrossSection(self.vertices,self.triangles)
        
        plane_normal=np.array(plane_normal)
        length=np.sqrt(plane_normal[0]**2+plane_normal[1]**2+plane_normal[2]**2)
        plane_normal=plane_normal/length
        
        sections=cross_section.get_multisections(plane_orig,plane_normal,num_slices)
        return sections
    
    def save_surface(self,surface,vertices,indices,normals,stitch=None):
        '''Saves Mesh as obj file that can be opend by external programs'''
        
        filename_surface = "surface_" + str(surface) + ".obj"
        if not stitch is None:
            filename_surface=str(stitch) + filename_surface
            
        fh=open(filename_surface,"w")
        
        fh.write('#%s vertices \n'%(self.num_columns))
        vv=0
        for v in vertices:
            vv+=1
            fh.write('v ' + str(int(v[0])) + ' ' + str(int(v[1])) + ' ' + str(int(v[2])) +'\n')
        print('v',vv)
        fh.write('#normals\n')
        nn=0
        for n in normals:
            nn+=1
            fh.write('vn ' + str(n[0]) + ' '+ str(n[1]) + ' ' + str(n[2]) + '\n')
        print('n',nn)
        fh.write('#faces\n')
        for i in indices:
            fh.write('f ' + str(i[0]+1) + ' ' + str(i[1]+1) + ' ' + str(i[2]+1) +'\n')
        fh.close
       