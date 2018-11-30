import matplotlib.pyplot as plt
from matplotlib import collections as mc
from collections import defaultdict
import pylab as pl
from scipy import ndimage
import numpy as np
from scipy.spatial import Delaunay

from spimagine import volfig, volshow
from spimagine import EllipsoidMesh, Mesh

from trimesh import base
#import pymesh

from Netsurface3d import NetSurf3d
from netsurface3d_radial import NetSurf3dRadial
from data4d import Data4d
from data4d_radial import Data4dRadial

from CrossSection import CrossSection
from stitch_surfaces import StitchSurfaces
    
    
class Visualization:
    '''
    Class for nicely displaying meshes that result from segmentation
      1) In 3D using spimagine
      2) as 2D cross-sections
    '''
    
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
    
    def __init__(self,data,images,mesh_positions=None):
        '''
        data  : list of Data4d or Data4DRadial instances used for segmentation
        images: unrotated, original input images
        mesh_positions: list, necessary for performing Boolean operations on meshes
                assigns to every mesh 'ol', 'ou', 'il' or 'iu'
        '''
        self.data = data
        self.images = images
        self.mesh_positions=mesh_positions
        K={}
        self.num_x={}
        self.num_y={}
        self.dx={}
        self.dy={}
        
        #taking segmentation parameters from respective Datad4(Radial)
        for d in range(3):
            if self.data[d] is not None:
                K[d], self.num_x[d], self.num_y[d] = self.data[d].K, images[0].shape[2]/self.data[d].dx, images[0].shape[1]/self.data[d].dy
                self.dx[d]=self.data[d].dx
                self.dy[d]=self.data[d].dy
        if not self.data[3] is None:
            K[3]=self.data[3].K
        self.order_parameters(self.num_x,self.num_y,K)
            
    def order_parameters(self,num_x,num_y,K):
        parameters={}
        for d in range(3):
            if not self.data[d] is None:
                if d==0:
                    parameters[0]=[num_x[0],num_y[0],K[0]]
                elif d==1:
                    parameters[1]=[num_x[1],K[1],num_y[1]]
                else:
                    parameters[2]=[K[2],num_y[2],num_x[2]]
        self.parameters = parameters
        
    '''
    3D Visualization with spimagine
    '''
        
    def show_frame( self, f, stackUnits=[1.,1.,1.], show_single=False, show_union=False, export=False, stitch=False ):        
        '''
        Displays 3D frame f with segmentation results in spimagine window
        [if show_union: will perform boolean operations 'difference' and 'intersection' like assigned by mesh_positions]
                    --> under construction
        if show_single: displays all individual segementation results in one Visualization, without performing operations
        '''
        assert f>=0 and f<len(self.images)

        self.current_frame = f    
      
        if show_union:
            self.show_single(f,stackUnits,raise_window=False,export=export,stitch=stitch)  
            self.show_union(f,stackUnits,raise_window=True,export=export)
            
        elif show_single and not show_union:
            self.show_single(f,stackUnits,raise_window=True,export=export,stitch=stitch)  
        
    def show_single(self,f,stackUnits,raise_window,export,stitch):
        '''
        Shows the 3D image (for time f)
        Calls for each method the create_surface_mesh() function in the corresponding Data4d(Radial)
        Displays the output meshes in the same spimagine window
        '''
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
                meshes=self.data[d].create_surface_mesh(f,k,facecolor=self.colors_diverse[d],export=export,stitch=stitch[d]) 
                for m in meshes:
                    print(m)
                    self.spimagine_single.glWidget.add_mesh(m)
                k+=1
                        
        if self.data[3] is not None:
            meshes=self.data[3].create_surface_mesh(f,facecolor=self.colors_diverse[3]) 
            for m in meshes:
                self.spimagine_single.glWidget.add_mesh(m) 
                
        return self.spimagine_single
                    
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
        '''
        shows cross-sections of num_slices parallel to plane with origin plane_orig and normal plane_normal
        num_slices: array of int numbers that indicate which slice-numbers along plane_normal to show
        '''
        
        #1. Determining sections through the meshes
        #sections: list of d entries, each entry a list of shape (s,n,m,2,3)
        #with:
        #       d: number of used methods (# of entries in self.data!=None)
        #       s: number of surfaces 
        #       n: number of slices (defined by num_slices)
        #       m: number of lines of shape (2,3) ... each line defined by start- and end-point of line with x,y,z-coordinates
        sections  = self.get_sections(f, plane_orig, plane_normal, num_slices)
        
        #direction: if plane_normal is [0,0,1] --> direction==2
        direction = self.get_direction(plane_normal)
        imgshape=np.array(self.images[f].shape[::-1])
        imgshape=np.delete(imgshape,direction)
        
        #frame: Number of slices
        frame=num_slices[-1]+1-num_slices[0]
        
        fig, ax = plt.subplots(frame,figsize=[10,80])
        
        #2. Sort lines of the different methods and surfaces by slice
        #and save line collections in lc
        lc={}
        for n, frames in enumerate(num_slices):
            lc[n]={}
            secs=defaultdict(list)
            for d, datas in enumerate(sections):
                for surface in datas:
                    if np.array(sections[d][surface][n][0]).shape[0]!=0:
                        secs[d].append(sections[d][surface][n][0])
                        
            #Process secs
            secs=self.process(secs,direction)

            colors=[ "limegreen","blue","gold","red"]
            used_colors=[]
            for dat in range(4):
                if self.data[dat] is not None:
                    used_colors.append(colors[dat])
            
            #use LineCollection to store lines in lc
            for l in secs:
                lc[n][l] = mc.LineCollection(secs[l],colors=used_colors[l],linewidths=2)
            
            #2D image with right dimensions
            if direction==0:
                img=self.images[f][:,:,frames]
            elif direction==1:
                img=self.images[f][:,frames,:]
            else:
                img=self.images[f][frames,:,:]
            
            #3. Show 2D plot(s)
            if frame==1:
                for l in secs:
                    ax.add_collection(lc[n][l])
                if not show_image is False:
                    ax.imshow(img,origin='lower')
                ax.axis([0,imgshape[0],0,imgshape[1]])
            elif frame>1:
                for l in secs:
                    ax[n].add_collection(lc[n][l])
                if not show_image is False:
                    ax[n].imshow(img,origin='lower')
                ax[n].axis([0,imgshape[0],0,imgshape[1]])
                    
    def get_sections(self, f, plane_orig,plane_normal,num_slices):
        '''
        Returns list containing lines that define the cross-sections of the meshes
        '''
        sections=[]
        k=0
        
        for d in range(4):
            if not self.data[d] is None:
                vertices,indices = self.data[d].give_surface_points(f,k) 
                cross_section=CrossSection(vertices,indices)
                
                #Normalizing plane_normal
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
        '''
        1. Reshaping each entry in diction, so that it is 3-dimensional
        2. Delete one dimension to get 2D coordinates
        3. Shifting coordinates by constant, to get useful results in cross-section
        '''
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
            array+=[0.5,2]
            new_diction[arr]=array
        return new_diction
    
    
    
    '''
    UNDER CONSTRUCTION: the part with the boolean operations
    '''
    
            
    def show_union(self,f,stackUnits,raise_window,export):
        if self.spimagine_union is None:
            self.spimagine_union = volshow(self.images[f], 
                                            stackUnits = stackUnits, raise_window=raise_window, autoscale=False)
        else:
            self.spimagine_union.glWidget.renderer.update_data(self.images[f])
            self.spimagine_union.glWidget.refresh()

        # remove all meshes (might eg exist from last call)
        self.hide_all_objects(single=False,union=True)

        vertices ={}
        indices ={}
        
        k=0
        for d in range(4):
            if not self.data[d] is None:
                vertices[d],indices[d] = self.data[d].give_surface_points(f,k)    
                vertices[d],indices[d] = np.squeeze(vertices[d]), np.squeeze(indices[d])
                print(vertices[d])
                k+=1
        
        image_shape = (self.images[0].shape[2],self.images[0].shape[1], self.images[0].shape[0]) 
        stitch = StitchSurfaces(vertices,indices,image_shape)

        vertices, indices = stitch.make_closed((self.dx,self.dy))

        print(np.array(vertices[0]).shape)
#         mesh=[]
                
#         for d in range(0,3):
#             if not self.data[d] is None:
#                 vertices[d]=np.squeeze(vertices[d])
#                 indices[d]=np.squeeze(indices[d])
#                 if len(np.array(vertices[d]).shape)==3:
#                     for surf in range(len(vertices[d])):
#                         mesh.append(self.create_mesh(vertices[d][surf],indices[d][surf],d, self.colors_diverse[d],export=export))
#                 else:
#                     mesh.append(self.create_mesh(vertices[d],indices[d],d, self.colors_diverse[d],export=export))
        
#         for m in mesh:
#             self.spimagine_union.glWidget.add_mesh(m)


        trimeshes={}
        for d in range(0,3):
            trimeshes[d]=[]
            if not self.data[d] is None:
                if not len(np.array(vertices[d]).shape)==2:
                    for surf in vertices[d]:
#                         if surf!=1:
                        print('jo')
                        print(surf)
                        trimeshes[d].append(base.Trimesh(vertices=np.array(vertices[d][surf]).astype(np.int),
                                                            faces=np.array(indices[d][surf]),process=False,validate=False))
                else:
                    print('jo?')
                    trimeshes[d].append(base.Trimesh(vertices=np.array(vertices[d]).astype(np.int),
                                                        faces=np.array(indices[d]),process=False,validate=False))
#                 else:
#                     print('jo?')
#                     trimeshes[d].append(base.Trimesh(vertices=vertices[d].astype(np.int),
#                                                         faces=indices[d],process=False,validate=False))
                    
#         union_mesh, additional = stitch.process(trimeshes,self.mesh_positions)


#         pymeshes={}
#         for d in range(0,3):
#             pymeshes[d]=[]
#             if not self.data[d] is None:
# #                 print(vertices[d])
# #                 vertices[d]=np.squeeze(vertices[d]).astype(np.int)
# #                 indices[d]=np.squeeze(indices[d])
# #                 if len(np.array(vertices[d]).shape)==2:                   
#                 for surf in vertices[d]:
# #                     print(surf)
#                     print(vertices[d][surf])
# #                         if surf==0:
#                     pymeshes[d].append(pymesh.form_mesh(vertices=np.array(vertices[d][surf]),
#                                                     faces=np.array(indices[d][surf])))
#                         else:
#                             pymeshes[d].append(pymesh.form_mesh(vertices=vertices[d][surf]+1,
#                                                             faces=indices[d][surf]))
#                 else:
#                     vertices = np.array(vertices[d]).astype(np.int)
#                     pymeshes[d].append(pymesh.form_mesh(vertices=vertices[d],
#                                                         faces=indices[d]))

#         new_vertices=np.array([[-10,-10,0],[-10,-10,20],[-10,130,20],[-10,130,0],[130,-10,0],[130,-10,20],[130,130,0],[130,130,20]])
#         new_indices=np.array([[0,1,3],[1,2,3],[0,1,4],[4,5,1],[0,4,3],[3,4,6],[2,3,6],[2,6,7],[1,2,5],[2,5,7],[4,5,7],[4,6,7]])
#         new_vertices1=new_vertices
#         new_vertices2=new_vertices+3
        
#         trimeshes={}
#         for d in range(0,4):
#             trimeshes[d]=[]
#         trimeshes[0].append(base.Trimesh(vertices=new_vertices1,faces=new_indices))
#         trimeshes[0].append(base.Trimesh(vertices=new_vertices2,faces=new_indices))
        
        print('before process')
        print(len(trimeshes[0]))
        union_mesh, additional = stitch.process(trimeshes,self.mesh_positions)
        print('after process')
#         for d in range(0,4):
#             pymeshes[d]=[]
#         pymeshes[0].append(pymesh.form_mesh(vertices=new_vertices1,faces=new_indices))
#         pymeshes[0].append(pymesh.form_mesh(vertices=new_vertices2,faces=new_indices))

#         union_mesh, additional = stitch.process(pymeshes,self.mesh_positions)
#         print(union_mesh)
        
        for d in range(0,3):
            if not self.data[d] is None:
                direction=d
                break
        
#         union_mesh = stitch.get_verts2(union_mesh)
        union_vertices, union_indices, union_normals = stitch.get_verts(union_mesh)
        union_mesh_spim = self.create_mesh(union_vertices, union_indices, union_normals, direction=direction, facecolor=self.colors_diverse[0])
        
        #ToDo: more than one additional
        if additional:
            add_vertices, add_indices, add_normals = stitch.get_verts(additional)
            add_mesh_spim = self.create_mesh(add_vertices, add_indices, add_normals, facecolor=self.colors_diverse[1])
                
        self.spimagine_union.glWidget.add_mesh(union_mesh_spim)
            
        return self.spimagine_union
    
    def create_mesh( self,  vertices=None, indices=None, normals=None, direction=0, facecolor=(1.,.3,.2), export=False):
        '''
        Generates one spimagine stitching Mesh that connects surface s and surface s+1
        -surface vertices and indices determined by stitch() in StitchSurfaces Class
        '''
        image_shape_xyz = (self.images[0].shape[2],self.images[0].shape[1], self.images[0].shape[0])   
        print(image_shape_xyz)
        
#         verts=np.zeros((indices.shape[0]*3,3))
        
#         k=0
        print(indices.shape)
        print(vertices.shape)
#         print(normals.shape)
#         for l in indices:
#             for m in l:
#                 verts[k]=vertices[m]
#                 verts[k]=self.norm_coords(verts[k],tuple(image_shape_xyz),direction)
#                 k+=1
                
        verts = vertices[indices].reshape(indices.shape[0]*3,3)
        for k in range(len(verts)):
            verts[k]=self.norm_coords(verts[k],tuple(image_shape_xyz),direction)
        
        ind=np.arange(0,3*indices.shape[0])
        
        if normals is None:
            mynormals=self.get_normals(indices,vertices)
            mynormals=mynormals.reshape(indices.shape[0]*3,3)
        else:            
            mynormals=normals[indices].reshape(indices.shape[0]*3,3)
            
        indices=ind.tolist()
        
        #TODO
        if export is not False:
            self.save_surface(s,stitch_verts,stitch_indices,mynormals,'union')

        return Mesh(vertices=verts, indices=indices, normals=mynormals, facecolor=facecolor, alpha=.5)        
     
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
        
    def norm_coords(self,cabs,pixelsizes,direction):
        """ 
        converts from absolute pixel location in image (x,y,z) to normalized [0,1] coordinates for spimagine meshes (z,y,x).
        performs translation operation for the mesh, so that every vertex is in the middle of the respective voxel
        """        
        cnorm = 2. * np.array(cabs, float) / np.array(pixelsizes) -1.  
        num_grid_z=min(self.parameters[direction][2],self.images[0].shape[0])
        cnorm[0]+=1/self.parameters[direction][0]
        cnorm[1]+=1/self.parameters[direction][1]
        cnorm[2]+=0.1/num_grid_z
#         num_grid_z=min(self.parameters[0][2],self.images[0].shape[0])
#         cnorm[0]+=1/self.parameters[0][0]
#         cnorm[1]+=1/self.parameters[0][1]
#         cnorm[2]+=5/num_grid_z
        
        return tuple(cnorm)
                                 
        
    def save_surface(self,surface,vertices,indices,normals,additional=None):
        '''Saves Mesh as obj file that can be opend by external programs'''
        
        filename_surface = "surface_" + str(surface) + ".obj"
        if not additional is None:
            filename_surface=str(additional) + filename_surface
            
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