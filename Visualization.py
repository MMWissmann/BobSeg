import matplotlib.pyplot as plt
from matplotlib import collections as mc
from collections import defaultdict
import pylab as pl
from scipy import ndimage
import numpy as np

from spimagine import volfig, volshow
from spimagine import EllipsoidMesh, Mesh

from Netsurface3d import NetSurf3d
from netsurface3d_radial import NetSurf3dRadial
from data4d import Data4d
from data4d_radial import Data4dRadial

from CrossSection import CrossSection
    
    
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
        
    '''
    3D Visualization with spimagine
    '''
        
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
                
    '''
    2D axis-parallel cross sections with matplotlib
    '''
    
    def show_sections(self,f=0,plane_orig=[0,0,0], plane_normal=[0,1,0], num_slices=0,show_image=True):
        
        sections  = self.get_sections(f, plane_orig, plane_normal, num_slices)
        direction = self.get_direction(plane_normal)
        
        imgshape=np.array(self.images[f].shape[::-1])
        imgshape=np.delete(imgshape,direction)
        
        frame=num_slices[-1]+1-num_slices[0]
        fig, ax = plt.subplots(frame,figsize=[10,80])
        
        lc={}
        k=0

        for frames in num_slices:
            lc[k]={}
            secs=defaultdict(list)
            l=0
            for datas in sections:
                for surface in datas:
                    if np.array(sections[l][surface][k][0]).shape[0]!=0:
                        secs[l].append(sections[l][surface][k][0])
                l+=1
            secs=self.process(secs,direction)

            colors=[ "limegreen","blue","gold","red"]
            used_colors=[]
            for d in range(4):
                if self.data[d] is not None:
                    used_colors.append(colors[d])
            
            for l in secs:
                lc[k][l] = mc.LineCollection(secs[l],colors=used_colors[l], linewidths=2)
            
            if direction==0:
                img=self.images[f][:,:,frames]
            elif direction==1:
                img=self.images[f][:,frames,:]
            else:
                img=self.images[f][frames,:,:]
            
            if frame==1:
                for l in secs:
                    ax.add_collection(lc[k][l])
                if not show_image is False:
                    ax.imshow(img,origin='lower')
                ax.axis([0,imgshape[0],0,imgshape[1]])
            elif frame>1:
                for l in secs:
                    ax[k].add_collection(lc[k][l])
                if not show_image is False:
                    ax[k].imshow(img,origin='lower')
                ax[k].axis([0,imgshape[0],0,imgshape[1]])
            k+=1
                    
    def get_sections(self, f, plane_orig,plane_normal,num_slices):
        sections=[]
        k=0
        
        for d in range(4):
            if not self.data[d] is None:
                vertices,indices = self.data[d].give_surface_points(f,k) 
                cross_section=CrossSection(vertices,indices)

                plane_normal=np.array(plane_normal)
                length=np.sqrt(plane_normal[0]**2+plane_normal[1]**2+plane_normal[2]**2)
                plane_normal=plane_normal/length

                sections.append((cross_section.get_multisections(plane_orig,plane_normal,num_slices) ) )
                k+=1
                
        return sections
    
    def get_direction(self,plane_normal):
        i=0
        for direc in plane_normal:
            if direc==1:
                direction=i
            i+=1
        return direction
    
    def process(self,diction,direction):
        new_diction={}
        for arr in diction:            
            array=np.array(diction[arr])
            assert len(array.shape)==4 or len(array.shape)==3 or len(array.shape)==1
            if len(array.shape)==4:
                n=array.shape[0]*array.shape[1]
                array=np.reshape(array,(n,2,3))
            elif len(array.shape)==1:
                new_array=np.concatenate([array[0],array[1]],axis=0)
                if len(array)>2:
                    for i in range(2,len(array)):
                        new_array=np.concatenate([new_array,array[i]],axis=0)
                array=new_array
            array=np.delete(array,direction,2)
            array+=[0,2]
            new_diction[arr]=array
        return new_diction