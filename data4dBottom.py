import numpy as np
import bresenham as bham
import maxflow
import math
from tifffile import imread, imsave
import _pickle as pickle

from spimagine import volfig, volshow
from spimagine import EllipsoidMesh, Mesh

from Netsurface3dFromBottom import NetSurfBottom

class Data4dBottom:
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
    
    def __init__( self, filenames, pixelsize=None, silent=True ):
        """
        Parameters:
            filenames   -  list of filenames (one per time point)
            pixelsize   -  calibration, eg. for volume computation
            silent      -  if True, no (debug/info) outputs will be printed on stdout
        """
        self.silent = silent
        self.filenames = filenames
        if not pixelsize is None: self.pixelsize = pixelsize
        
        # load images
        self.load_from_files( self.filenames )

    # ***********************************************************************************************
    # *** SEGMENTATION STUFF *** SEGMENTATION STUFF *** SEGMENTATION STUFF *** SEGMENTATION STUFF ***
    # ***********************************************************************************************

    def set_seg_params( self, K, max_delta_k, divx, divy ):
        self.K = K
        self.max_delta_k = max_delta_k
        self.divx = divx
        self.divy = divy
        
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
        
    def add_object_at ( self, oid, min_rs, max_rs, frame, frame_to=None, segment_it=False ):
        """
        Makes a given (already added) object exist at a frame (or a sequence of consecutive frames).
        Parameters:
            oid         -  the object id as returned by 'add_object'
            min_rs      -  value defining min # of pixels from bottom to look for the object surface
            max_rs      -  value defining max # of pixels from bottom to look for the object surface
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
        
        for i in range(frame,frame_to+1):
            self.object_min_surf_dist[oid][i] = (min_rs)
            self.object_max_surf_dist[oid][i] = max_rs        
            self.object_visibility[oid][i] = True
           
            if not self.silent:
                print('Added appearance for "'+str(self.object_names[oid])+ \
                      '" in frame', i)
            if segment_it: 
                self.segment_frame( oid, i )
    
    def segment( self, oids=None ):
        """
        Segments all added object occurances using NetSurf3dFromBottom.
        If oids is not given (None) this will be done for ALL objects at ALL time points.
        """
        if oids is None:
            oids = range(len(self.object_names)) # all ever added
        for i,oid in enumerate(oids):
            if not self.silent:
                print('Working on "'+str(self.object_names[oid])+ \
                      '" (object', i+1, 'of', len(oids),')...')
            self.netsurfs[oid] = [None] * len(self.images)
            for f, visible in enumerate(self.object_visibility[oid]):
                if visible:
                    if not self.silent:
                        print('   Segmenting in frame '+str(f)+'...')
                    self.segment_frame( oid, f )
                    self.object_volumes[oid][f] = self.netsurfs[oid][f].get_volume( self.pixelsize )
                    if not self.silent:
                        print('      Volume: ', self.object_volumes[oid][f])

    def segment_frame( self, oid, f ):
        """
        Segments object oid in frame f.
        """
        assert oid>=0
        assert oid<len(self.object_names)
        assert f>=0
        assert f<len(self.images)
        
        try:
            self.netsurfs[oid][f] = None
        except:
            self.netsurfs[oid] = [None] * len(self.images)
        
        self.netsurfs[oid][f] = NetSurfBottom(K=self.K, max_delta_k=self.max_delta_k, divx=self.divx, divy=self.divy)
        optimum = self.netsurfs[oid][f].apply_to(self.images[f], 
                                                 self.object_max_surf_dist[oid][f], 
                                                 self.object_min_surf_dist[oid][f])
        if not self.silent:
            print('      Optimum energy: ', optimum)
            ins, outs = self.netsurfs[oid][f].get_counts()
            print('      Nodes in/out: ', ins, outs)
            
    # ***************************************************************************************************
    # *** TRACKING&REFINEMENT *** TRACKING&REFINEMENT *** TRACKING&REFINEMENT *** TRACKING&REFINEMENT ***
    # ***************************************************************************************************
    
    def track( self, oid, seed_frame, target_frames ):
        """
        For the object with the given id this function tries to fill in missing frames.
        Note: this will only work if the object seed in seed_frame lies within the desired object in
        the first target_frames and the re-evaluated center in each consecutive target frame (iterated).
        Parameters:
            oid           -  object id that should be tracked
            seed_frame    -  frame id that was previously seeded (using add_object_at)
            target_frame  -  list of frame ids the object should be tracked at
         """
        assert oid>=0
        assert oid<=len(self.object_names)
        print(oid)
        print(self.object_min_surf_dist[oid][seed_frame])

        min_rs = self.object_min_surf_dist[oid][seed_frame]
        max_rs = self.object_max_surf_dist[oid][seed_frame]
        for f in target_frames:
            self.add_object_at( oid, min_rs, max_rs, f )
            self.segment_frame( oid, f )

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

    def load_from_files( self, filenames ):
        self.images = [None]*len(filenames)
        for i in range(len(filenames)):
            self.images[i] = imread(filenames[i])
            if not self.silent:
                print('Dimensions of frame', i, ': ', self.images[i].shape)
            
    # ***************************************************************************************************
    # *** VISUALISATION STUFF *** VISUALISATION STUFF *** VISUALISATION STUFF *** VISUALISATION STUFF ***
    # ***************************************************************************************************
            
        
    def get_result_polygone( self, oid, frame ):
        points=[]
        col_vectors = self.netsurfs[oid][frame].col_vectors
        netsurf = self.netsurfs[oid][frame]
        for i in range( len(col_vectors) ):
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
    
    def give_surface_points(self,f):
        x = np.zeros((20000,3))
        for oid in range(len(self.object_names)):
            netsurf = self.netsurfs[oid][f]
            x= netsurf.give_surface_points()
        return x
    
    def show_frame( self, f, show_surfaces=False, show_centers=False, stackUnits=[1.,1.,1.], raise_window=True ):
        assert f>=0 and f<len(self.images)
        
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
                if show_surfaces: self.spimagine.glWidget.add_mesh( 
                        netsurf.create_surface_mesh( facecolor=self.colors_diverse[0]) )
        return self.spimagine
    
    def hide_all_objects( self ):
        while len(self.spimagine.glWidget.meshes)>0:
            self.spimagine.glWidget.meshes.pop(0)
        self.spimagine.glWidget.refresh()

    def show_objects( self, oids, show_surfaces=True, show_centers=True, colors=None ):
        assert not self.current_frame is None
        if colors is None:
            colors = self.colors_diverse
            
        i = 0
        for oid in oids:
            assert oid>=0 and oid<len(self.object_names)
            netsurf = self.netsurfs[oid][self.current_frame]
            if not netsurf is None:
                if show_centers:  self.spimagine.glWidget.add_mesh( 
                        netsurf.create_center_mesh( facecolor=colors[i%len(colors)]) )
                if show_surfaces: self.spimagine.glWidget.add_mesh( 
                        netsurf.create_surface_mesh( facecolor=colors[i%len(colors)]) )
                i += 1
                
    def save_current_visualization( self, filename ):
        self.spimagine.saveFrame( filename )