import matplotlib.pyplot as plt
from matplotlib import collections as mc
import pylab as pl
from scipy import ndimage

from spimagine import volfig, volshow
from spimagine import EllipsoidMesh, Mesh

from Netsurface3d import NetSurf3d
from netsurface3d_radial import NetSurf3dRadial
from data4d import Data4d
from data4d_radial import Data4dRadial
    
    
class Visualization:
    
    spimagine_single=None
    spimagine_union=None
    object_names=[]
    
    colors_grey = [(1.-.15*i,1.-.15*i,1.-.15*i) for i in range(6)]
    colors_red = [(1.,.2,.2*i) for i in range(6)]
    colors_gold = [(1.,.8,.15*i) for i in range(6)]
    colors_yellow = [(1.,1.,.9-.15*i) for i in range(6)]
    colors_green = [(.45,1.,.4+.1*i) for i in range(6)]
    colors_blue = [(.4,1.,1.3+1*i) for i in range(6)]
    colors_darkblue = [(.1,.3,1.0-.1*i) for i in range(6)]
    colors_diverse = [ colors_green[0], colors_blue[0], colors_gold[0], colors_red[0], colors_yellow[0],colors_grey[1] ]
    
    def __init__(self,data,images):
        self.data = data
        self.images = images
        
    def show_frame( self, f, show_union=False, show_single=False, 
                   stackUnits=[1.,1.,1.], raise_window=True, export=False, stitch=False ):
        
                        
        assert f>=0 and f<len(self.images)

        self.current_frame = f    
      
        if show_union:
            self.show_union(f,stackUnits,raise_window,export)
            
        if show_single:
            self.show_single(f,stackUnits,raise_window,export,stitch)  
        
    def show_single(self,f,stackUnits,raise_window,export,stitch):
        if self.spimagine_single is None:
            self.spimagine_single = volshow(self.images[f], stackUnits = stackUnits, 
                                            raise_window=raise_window, autoscale=False)
        else:
            self.spimagine_single.glWidget.renderer.update_data(self.images[f])
            self.spimagine_single.glWidget.refresh()

        # remove all meshes (might eg exist from last call)
        self.hide_all_objects(single=True,union=False)
        
        k=0
        for d in range(3):
            meshes=None
            if self.data[d] is not None:
                meshes=self.data[d].create_surface_mesh(f,k,facecolor=self.colors_diverse[d],export=export,stitch=stitch) 
                for m in meshes:
                    print('m',m)
                    self.spimagine_single.glWidget.add_mesh(m)
                k+=1
                        
        if self.data[3] is not None:
            meshes=self.data[3].create_surface_mesh(f,facecolor=self.colors_diverse[3]) 
            for m in meshes:
                self.spimagine_single.glWidget.add_mesh(m) 
                
        return self.spimagine_single
        
        
    def show_union(self,f,stackUnits,raise_window,export):
        if self.spimagine_union is None:
            self.spimagine_union = volshow(self.images[f], 
                                            stackUnits = stackUnits, raise_window=raise_window, autoscale=False)
        else:
            self.spimagine_union.glWidget.renderer.update_data(self.images[f])
            self.spimagine_union.glWidget.refresh()

        # remove all meshes (might eg exist from last call)
        self.hide_all_objects(single=False,union=True)
             
        
    def hide_all_objects( self, single, union ):
        if single is True:
            while len(self.spimagine_single.glWidget.meshes)>0:
                self.spimagine_single.glWidget.meshes.pop(0)
            self.spimagine_single.glWidget.refresh()
        if union is True:
            while len(self.spimagine_union.glWidget.meshes)>0:
                self.spimagine_union.glWidget.meshes.pop(0)
                self.spimagine_union.glWidget.refresh()

    def save_current_visualization( self, filename ):
        self.spimagine.saveFrame( filename )