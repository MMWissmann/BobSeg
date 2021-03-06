import numpy as np
import bresenham as bham
import maxflow
import math
from tifffile import imread, imsave
import _pickle as pickle
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import pylab as pl
from scipy import ndimage

from spimagine import volfig, volshow
from spimagine import EllipsoidMesh, Mesh

from Netsurface3d import NetSurf3d

class Data4d:
    """
    Implements a container to hold 3d+t (4d) time-lapse datasets.
    Time points in such datasets can conveniently be segmented via NetSurface3d and 
    visualized in Spimagine.
    """
    
    silent = True
    filenames = []
    images = []
    pixelsize=(1.,1.,1.)
    object_names = []
    object_visibility = {}
    object_seedpoints = {}
    object_volumes = {}
    object_min_surf_dist = {}
    object_max_surf_dist = {}

    netsurfs = {}
    spimagine = None
    current_fame = None
    mask = None
    
    # global segmentation parameters (for NetSurf3d)
    # (set from outside using method 'set_seg_params')
    K = 30
    max_delta_k = 4
    
    colors_grey = [(1.-.15*i,1.-.15*i,1.-.15*i) for i in range(6)]
    colors_red = [(1.,.2,.2*i) for i in range(6)]
    colors_gold = [(1.,.8,.15*i) for i in range(6)]
    colors_yellow = [(1.,1.,.9-.15*i) for i in range(6)]
    colors_green = [(.45,1.,.4+.1*i) for i in range(6)]
    colors_blue = [(.4,1.,1.3+1*i) for i in range(6)]
    colors_darkblue = [(.1,.3,1.0-.1*i) for i in range(6)]
    colors_diverse = [ colors_green[0], colors_red[0], colors_blue[0], colors_gold[0], colors_yellow[0],colors_grey[1] ]
    
    def __init__( self, filenames, axis=0, filenames_mask=None, pixelsize=None, silent=True, plane=None ):
        """
        Parameters:
            filenames   -  list of filenames (one per time point)
            axis   -  searching for surfaces along z,x, or y-axis
            filenames_mask- list of filenames of 2d mask images
            pixelsize   -  calibration, eg. for volume computation
            silent      -  if True, no (debug/info) outputs will be printed on stdout
            plane       -  if True, additional plane on top and bottom will be added with average intensity of whole image
        """
        self.silent = silent
        self.axis = axis
        self.filenames = filenames
        print('filenames_mask',filenames_mask)
        NoneType=type(None)
        if not isinstance(pixelsize,NoneType): self.pixelsize = pixelsize
        
        if self.axis==0:
            axis='z'
        if self.axis==1:
            axis='y'
        if self.axis==2:
            axis='x'
        
        # load images
        self.load_from_files( self.filenames, axis )
        if not isinstance(filenames_mask,NoneType): 
            self.filenames_mask = filenames_mask
            self.load_from_files_mask( self.filenames_mask )
        if not plane is None: self.add_plane()

    # ***********************************************************************************************
    # *** SEGMENTATION STUFF *** SEGMENTATION STUFF *** SEGMENTATION STUFF *** SEGMENTATION STUFF ***
    # ***********************************************************************************************

    def set_seg_params( self, K, max_delta_k, dx, dy, surfaces, min_dist, max_dist):
        '''
        Set parameters for building the graphical model for segmentation
            K           -  how many sample points per column
            max_delta_k -  maximum column height change between neighbors (as defined by adjacency)
            dx/dy       -  divisor: choose only every d-th pixel to create a column on base graph, for x and y
            surfaces    -  number of surfaces to be detected
            min/max_dist-  for more than 1 surface: minimum and maximum intersurface distances !in terms of K!
        '''
        self.K = K
        self.max_delta_k = max_delta_k
        self.dx = dx
        self.dy = dy
        self.min_dist=min_dist
        self.max_dist=max_dist
        self.surfaces=surfaces
        
    def init_object( self, name ):
        """
        Adds an (empty) object definition to this dataset.
        Returns the id of the added object.
        """
        oid = self.get_object_id( name )
        if oid == -1: # not found
            oid = len(self.object_names)
            self.object_names.append(name)
        self.object_visibility[oid] = [False] * len(self.images)
        self.object_seedpoints[oid] = [None] * len(self.images)
        self.object_min_surf_dist[oid] = [(0)] * len(self.images)
        self.object_max_surf_dist[oid] = [(100)] * len(self.images)
        self.object_volumes[oid] = [0] * len(self.images)
        self.netsurfs[oid] = [None] * len(self.images)
        return oid
    
    def get_object_id( self, name ):
        for i,n in enumerate(self.object_names):
            if name == n:
                return i
        return -1
        
    def add_object_at ( self, oid, min_surf_dist, max_surf_dist, 
                       frame, frame_to=None, segment_it=False, plot_base_graph=False ):
        """
        Makes a given (already added) object exist at a frame (or a sequence of consecutive frames).
        Parameters:
            oid         -  the object id as returned by 'add_object'
            min_surf_dist-  value defining min # of pixels from bottom to look for the object surface
            max_surf_dist-  value defining max # of pixels from bottom to look for the object surface
            frame       -  the frame index at which the object occurs (is visible)       
            frame_to    -  if given and >frame, all frames in [frame,frame_end] will me marked to contain this object
        """
        if frame_to is None: 
            frame_to = frame
            
        assert frame >= 0
        assert frame < len(self.images)
        assert frame_to >= 0
        assert frame_to < len(self.images)
        assert frame <= frame_to
        
        if max_surf_dist == 0:
            max_surf_dist = self.images[0].shape[0]
        
        for i in range(frame,frame_to+1):
            self.object_min_surf_dist[oid][i] = (min_surf_dist)
            self.object_max_surf_dist[oid][i] = max_surf_dist        
            self.object_visibility[oid][i] = True
           
            if not self.silent:
                print('Added appearance for "'+str(self.object_names[oid])+ \
                      '" in frame', i)
            if segment_it: 
                self.segment_frame( oid, i, plot_base_graph )
    
    def segment_frame( self, oid, f, plot_base_graph ):
        """
        Segments object oid in frame f.
        """
        assert oid>=0
        assert oid<len(self.object_names)
        assert f>=0
        assert f<len(self.images)
        if self.mask is None: mask=None
        else: mask = self.mask

        
        try:
            self.netsurfs[oid][f] = None
        except:
            self.netsurfs[oid] = [None] * len(self.images)
        
        self.netsurfs[oid][f] = NetSurf3d(K=self.K, max_delta_k=self.max_delta_k, dx=self.dx, dy=self.dy, surfaces=self.surfaces, min_dist=self.min_dist, max_dist=self.max_dist, axis=self.axis)
        optimum = self.netsurfs[oid][f].apply_to(self.images[f], 
                                                 self.object_max_surf_dist[oid][f], 
                                                 self.object_min_surf_dist[oid][f],
                                                 mask,
                                                 plot_base_graph)
        if not self.silent:
            print('      Optimum energy: ', optimum)
            ins, outs = self.netsurfs[oid][f].get_counts()
            print('      Nodes in/out: ', ins, outs)
            

    # *****************************************************************************************************
    # *** SAVE&LOAD *** SAVE&LOAD *** SAVE&LOAD *** SAVE&LOAD *** SAVE&LOAD *** SAVE&LOAD *** SAVE&LOAD ***
    # *****************************************************************************************************
    
    def save( self, filename ):
        dictDataStorage = {
            'silent'               : self.silent,
            'filenames'            : self.filenames,
            'object_names'         : self.object_names,
            'object_visibility'    : self.object_visibility,
            'object_volumes'       : self.object_volumes,
            'object_min_surf_dist' : self.object_min_surf_dist,
            'object_max_surf_dist' : self.object_max_surf_dist,
            'current_fame'         : self.current_fame,
            'K'                    : self.K,
            'max_delta_k'          : self.max_delta_k,
        }
        with open(filename,'wb') as f:
            pickle.dump(dictDataStorage,f)

    def load( self, filename, compute_netsurfs=True ):
        with open(filename,'r') as f:
            dictDataStorage = pickle.load(f)

        self.silent = dictDataStorage['silent']
        self.filenames = dictDataStorage['filenames']
        self.object_names = dictDataStorage['object_names']
        self.object_visibility = dictDataStorage['object_visibility']
        self.object_volumes = dictDataStorage['object_volumes']
        self.object_min_surf_dist = dictDataStorage['object_min_surf_dist']
        self.object_max_surf_dist = dictDataStorage['object_max_surf_dist']
        self.current_fame = dictDataStorage['current_fame']
        self.K = dictDataStorage['K']
        self.max_delta_k = dictDataStorage['max_delta_k']
        
        self.load_sphere_sampling()
        self.load_from_files( self.filenames ) # load the raw images from file too!!!
        if compute_netsurfs: self.segment()

    def load_from_files( self, filenames, axis ):
        self.images = [None]*len(filenames)
        swap=0
        if axis=='x':
            swap=2
        elif axis=='y':
            swap=1
        for i in range(len(filenames)):
            self.images[i] = imread(filenames[i])
            if not self.silent:
                print('Dimensions (z,y,x) of frame', i, ': ', self.images[i].shape)
                print('Searching for surface(s) along %s direction'%axis)
            if swap != 0:
                print('Changing %s and z axis'%axis)
                self.images[i] = np.swapaxes(self.images[i],0,swap)
                print('New dimensions (z,y,x) of frame', i, ':', self.images[i].shape)
                            
    def load_from_files_mask( self, mask_names ):
#         self.mask = [None]*len(mask_names)
#         for i in range(len(mask_names)):
#             image_obj = Image.open(mask_names[i])
#             self.mask[i] = imread(mask_names[i])
        image_obj = Image.open(mask_names)
        self.mask = imread(mask_names)
        if not self.silent:
#             print('Dimensions (y,x) of mask', i, ': ', self.mask[i].shape)
            print('Dimensions (y,x) of mask', self.mask.shape)
            
    def add_plane(self):
        av=np.average(self.images[0])
        plane=np.full((self.images[0].shape[1],self.images[0].shape[2]),av)
        self.images[0]=np.insert(self.images[0],0,plane,axis=0)
        self.images[0]=np.append(self.images[0],[plane],axis=0)
        print(self.images[0].shape)
        
    # ***************************************************************************************************
    # *** VISUALISATION STUFF *** VISUALISATION STUFF *** VISUALISATION STUFF *** VISUALISATION STUFF ***
    # ***************************************************************************************************
            
        
    def get_result_polygone( self, oid, frame ):
        points=[]
        base_coords = self.netsurfs[oid][frame].base_coords
        netsurf = self.netsurfs[oid][frame]
        for i in range( len(base_coords) ):
            points.append( netsurf.get_surface_point(i) )
        return points
    
    def create_segmentation_image(self, dont_use_2dt=False):
        segimgs = np.zeros_like(self.images)
        for f in range(len(self.images)):
            vis = np.zeros((np.shape(segimgs)[1],np.shape(segimgs)[2],3), np.uint8)
            # retrieve polygones
            polygones = []
            for oid in range(len(self.object_names)):
                assert oid>=0 and oid<len(self.object_names)      
                polygones.append( self.get_result_polygone(oid,f) )

            # draw polygones
            for polygone in polygones:
                cv2.polylines(vis, np.array([polygone], 'int32'), 1, (128,128,128), 2)
                cv2.polylines(vis, np.array([polygone], 'int32'), 1, (255,255,255), 1)


            segimgs[f] = vis[:,:,0]
        return segimgs
    
    def give_surface_points(self,f,oid):
        '''
        Returns vertices & faces of the triangulated mesh representing the segmentation result
        Works only after create_surface_mesh in Netsurface3d has run (that's where the mesh generation happens)
        '''
        x = []
        i=[]
        netsurf = self.netsurfs[oid][f]
        x.append(netsurf.vertices)
        i.append(netsurf.triangles)
        print(np.array(x).shape)
        return np.array(x),np.array(i)
    
    def get_segment(self,f,column_id):
        for oid in range(len(self.object_names)):
            netsurf=self.netsurfs[oid][f]
            print(column_id)
            segment=np.zeros(netsurf.K)
            for k in range(netsurf.K):
                s=netsurf.g.get_segment(column_id*netsurf.K+k)
                segment[k]= s
        return segment
    
    def show_frame( self, f, show_surfaces=False, show_centers=False, stackUnits=[1.,1.,1.], raise_window=True, export=False, stitch=False ):
        assert f>=0 and f<len(self.images)
        
        if self.axis != 0:
            self.images[f] = np.swapaxes(self.images[f],0,self.axis)
        
        self.current_frame = f
        if self.spimagine is None:
            self.spimagine = volshow(self.images[f], stackUnits = stackUnits, raise_window=raise_window, autoscale=False)
        else:
            self.spimagine.glWidget.renderer.update_data(self.images[f])
            self.spimagine.glWidget.refresh()
        
        # remove all meshes (might eg exist from last call)
        self.hide_all_objects()
        
        for oid in range(len(self.object_names)):
            netsurf = self.netsurfs[oid][f]
            if not netsurf is None:
                if show_surfaces: 
                    for s in range(self.surfaces):
                        self.spimagine.glWidget.add_mesh(netsurf.create_surface_mesh( s,facecolor=self.colors_diverse[0],export=export) )
                if not stitch is False:
                    for s in range(self.surfaces-1):
                        print('stitching surfaces: ', s, s+1)
                        self.spimagine.glWidget.add_mesh(netsurf.create_stitching_mesh( s,facecolor=self.colors_diverse[0],export=export) )
        return self.spimagine
    
    def create_surface_mesh(self,f,oid,facecolor,export=False,stitch=False):
        mesh=[]
        netsurf = self.netsurfs[oid][f]
        for s in range(self.surfaces):
            mesh.append(netsurf.create_surface_mesh( s,facecolor,export))
        if not stitch is False:
            assert self.surfaces > 1
            for s in range(self.surfaces-1):
                print('stitching surfaces: ', s, s+1)
                mesh.append(netsurf.create_stitching_mesh( s,facecolor,export) )
        return mesh
                
    def show_sections(self,f,plane_orig,plane_normal,num_slices,show_image=False):
        '''
        Displays cross sections through resulted segmentation meshes in the plane defined by origin and normal
        '''
        for oid in range(len(self.object_names)):
            netsurf = self.netsurfs[oid][f]
            d=self.get_direction(plane_normal)
            
            if not netsurf is None:
                secs=netsurf.show_sections(plane_orig,plane_normal,num_slices)
                print('in data show')
#                 print(secs[0].shape)
#                 print(secs)
                
                imgshape=np.array(self.images[f].shape[::-1])
                imgshape=np.delete(imgshape,d)
                frame=num_slices[-1]+1-num_slices[0]
                fig, ax = plt.subplots(frame,figsize=[10,80])              

            lc={}
            k=0
            for frames in num_slices:
                sec=[]
                for surface in secs:
                    sec.append(secs[surface][k][0])
                sec=self.process(sec,d)

                print('frame',frames)

                lc[k] = mc.LineCollection(sec, linewidths=2)

                if d==0:
                    img=self.images[f][:,:,frames]
                elif d==1:
                    img=self.images[f][:,frames,:]
                else:
                    img=self.images[f][frames,:,:]

                if frame==1:
                    ax.add_collection(lc[k])
                    if not show_image is False:
                        ax.imshow(img,origin='lower')
                    ax.axis([0,imgshape[0],0,imgshape[1]])
                elif frame>1:
                    ax[k].add_collection(lc[k])
                    if not show_image is False:
                        ax[k].imshow(img,origin='lower')
                    ax[k].axis([0,imgshape[0],0,imgshape[1]])
                k+=1
    
    def get_direction(self,plane_normal):
        i=0
        for direc in plane_normal:
            if direc==1:
                direction=i
            i+=1
        return direction
    
    def process(self,array,direction):
        '''
        Calibration of coordinates for displaying cross sections
        '''
        array=np.array(array)
        assert len(array.shape)==4 or len(array.shape)==3
        if len(array.shape)==4:
            n=array.shape[0]*array.shape[1]
            array=np.reshape(array,(n,2,3))
        array=np.delete(array,direction,2)
        array+=[0,2]
        return array

    def hide_all_objects( self ):
        while len(self.spimagine.glWidget.meshes)>0:
            self.spimagine.glWidget.meshes.pop(0)
        self.spimagine.glWidget.refresh()

    def show_objects( self, oids, show_surfaces=True, show_centers=False, colors=None ):
        assert not self.current_frame is None
        if colors is None:
            colors = self.colors_diverse
            
        i = 0
        for oid in oids:
            assert oid>=0 and oid<len(self.object_names)
            netsurf = self.netsurfs[oid][self.current_frame]
            if not netsurf is None:
                for s in range(self.surfaces):
                    if show_centers:  self.spimagine.glWidget.add_mesh( 
                            netsurf.create_center_mesh( facecolor=colors[i%len(colors)]) )
                    if show_surfaces: self.spimagine.glWidget.add_mesh( 
                            netsurf.create_surface_mesh( s,facecolor=colors[i%len(colors)]) )
                    i += 1
                
    def save_current_visualization( self, filename ):
        self.spimagine.saveFrame( filename )