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
from stitch_surfaces import StitchSurfaces
    
    
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
        K={}
        num_x={}
        num_y={}
        for d in range(3):
            if self.data[d] is not None:
                K[d], num_x[d], num_y[d] = self.data[d].K, images[0].shape[2]/self.data[d].dx, images[0].shape[1]/self.data[d].dy
        if not self.data[3] is None:
            K[3]=self.data[3].K
        self.order_parameters(num_x,num_y,K)
    
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
                        
        assert f>=0 and f<len(self.images)

        self.current_frame = f    
      
        if show_union:
            if not show_single:
                self.show_single(f,stackUnits,raise_window=False,export=export,stitch=stitch)
            elif show_single:
                print('here')
                self.show_single(f,stackUnits,raise_window=True,export=export,stitch=stitch)
                print('done')
            self.show_union(f,stackUnits,raise_window=True,export=export)
            
        elif show_single and not show_union:
            self.show_single(f,stackUnits,raise_window=True,export=export,stitch=stitch)  
        
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
                    print(m)
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
        
        vertices, indices = self.remove_indices(f)
        
        print(vertices[0].shape, indices[0].shape)
        print(vertices[0][0][0])
        print(vertices[1].shape, indices[1].shape)
        
        assert len(vertices[0].shape) == len(indices[0].shape)
        
        stitch = StitchSurfaces(vertices,indices)
        
        connect_vertices={}
        connect_indices={}

        for d in range(1,3):
            if not self.data[d] is None:
                connect_vertices[d], connect_indices[d] = stitch.connect(0,d)
        
        mesh=[]
        
        if len(vertices[0].shape)==3:
            for surf in range(len(vertices[0])):
                main_mesh = self.create_mesh(vertices[0][surf],indices[0][surf],0, self.colors_diverse[0],export=export)
                mesh.append(main_mesh)
        else:
            main_mesh = self.create_mesh(vertices[0],indices[0],0, self.colors_diverse[0],export=export)
            mesh.append(main_mesh)
                
        for d in range(1,3):
            if not self.data[d] is None:
                mesh.append(self.create_mesh(vertices[d],indices[d],d, self.colors_diverse[d],export=export))
                assert not connect_vertices[d] is None
                mesh.append(self.create_mesh(connect_vertices[d],connect_indices[d],d,self.colors_grey[0],export=export)) 
        
        for m in mesh:
            self.spimagine_union.glWidget.add_mesh(m)
            
        return self.spimagine_union
            
    def hide_all_objects( self, single, union ):
        if single is True:
            while len(self.spimagine_single.glWidget.meshes)>0:
                self.spimagine_single.glWidget.meshes.pop(0)
            self.spimagine_single.glWidget.refresh()
        if union is True:
            while len(self.spimagine_union.glWidget.meshes)>0:
                self.spimagine_union.glWidget.meshes.pop(0)
                self.spimagine_union.glWidget.refresh()
                                
    def remove_indices(self,f):
        vertices ={}
        indices ={}
        
        k=0
        for d in range(4):
            if not self.data[d] is None:
                vertices[d],indices[d] = self.data[d].give_surface_points(f,k)    
                vertices[d],indices[d] = np.squeeze(vertices[d]), np.squeeze(indices[d])
#                 if d==1 or d==2:
#                     print('in transform')
#                     print(vertices[d][0])
#                     axis=abs(d-2)        #from [012] --> [210]
#                     vertices[d][:,[2,axis]] = vertices[d][:,[axis,2]]
#                     print(vertices[d][0])
                k+=1
        
        several_main_surfaces=False
        if len(vertices[0].shape)==3:
            several_main_surfaces=True
        
#         main_verts=[]
        main_inds=[]
        index={}
        for d in range(1,3):
            if not self.data[d] is None:
                maxi=int(max(vertices[d][:,d]))
                mini=int(min(vertices[d][:,d]))
                print(mini)
                print(maxi)
                if several_main_surfaces:
                    print(len(vertices[0]))
                    for surfs in range(len(vertices[0])):
                        index[surfs]=[]
                        mains=np.array(vertices[0][surfs])
                        for vertex in range(mini-1,maxi+1):
                            inds, mains = self.find_indices(float(vertex),mains,d)
                            if len(inds):
                                index[surfs]=np.append(index[surfs],inds)
#                         main_verts.append(mains)
#                     vertices[0]=np.array(main_verts)
                else:
                    for vertex in range(mini-1,maxi+1):
#                         index, vertices[0] = self.find_indices(float(vertex),vertices[0],d)
                        index, etwas = self.find_indices(float(vertex),vertices[0],d)
                        if len(inds):
                            index=np.append(index,inds)
                if several_main_surfaces:
                    for surfs in range(len(index)):
                        if len(indices[0].shape)==2:
                            mains=np.array(indices[0])
                        elif len(indices[0].shape)==3:
                            mains=np.array(indices[0][surfs])
                        for ind in index[surfs]:
#                             print('indices here',ind)
                            for dim in range(3):
                                i, mains = self.find_indices(ind,mains,dim)
#                                 print(mains.shape)
                        main_inds.append(mains)
                    indices[0]=np.array(main_inds)
                else:
                    for ind in index:
                        for dim in range(3):
                            i, indices[0] = self.find_indices(ind,indices[0],dim)
                            
        vertices[0],indices[0] = self.remove_vertices(vertices[0],indices[0],several_main_surfaces)
        
        return vertices, indices
                                  
    def find_indices(self,value, my_list,d):
        index=[]
        while value in my_list[:,d]:
            value_index = np.where(my_list[:,d]==value)
            index.append(value_index)
            my_list=np.delete(my_list,value_index,0)
#             my_list[value_index]=[0,0,0]
        
        return index, my_list
    
    def remove_vertices(self,vertices,indices,several_surfaces=False):
        ind=[]
        verts=[]
        if several_surfaces:
            for surf in range(len(vertices)):
                num_inds=3*indices[surf].shape[0] 
                print('num_inds',num_inds)
                ind.append(np.arange(0,num_inds).reshape(indices[surf].shape[0],3))
                vertis=vertices[surf][indices[surf]]
                verts.append(vertis.reshape(num_inds,3))
#                 if surf==0:
#                     inds=[ind[surf]]
#                 elif surf>0:
#                     inds=np.append(inds,[ind[surf]],axis=0)
                    
        else:
            num_inds=3*indices.shape[0]        
            ind=np.arange(0,num_inds)
            verts=vertices[indices].reshape(num_inds,3)
            
        return np.array(verts), np.array(ind)
            
    def create_mesh( self,  vertices=None, indices=None, direction=0, facecolor=(1.,.3,.2), export=False):
        '''
        Generates one spimagine stitching Mesh that connects surface s and surface s+1
        -surface vertices and indices determined by stitch() in StitchSurfaces Class
        '''
        image_shape_xyz = (self.images[0].shape[2],self.images[0].shape[1], self.images[0].shape[0])   
        print(image_shape_xyz)
        
        verts=np.zeros((indices.shape[0]*3,3))
        
        k=0
        print(indices.shape)
        print(vertices.shape)
        for l in indices:
            for m in l:
                verts[k]=vertices[m]
                verts[k]=self.norm_coords(verts[k],tuple(image_shape_xyz),direction)
                k+=1
        
        ind=np.arange(0,3*indices.shape[0])
        mynormals=self.get_normals(indices,vertices)
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
#         num_grid_z=min(self.parameters[direction][2],self.images[0].shape[0])
#         cnorm[0]+=1/self.parameters[direction][0]
#         cnorm[1]+=1/self.parameters[direction][1]
#         cnorm[2]+=3/num_grid_z
        num_grid_z=min(self.parameters[0][2],self.images[0].shape[0])
        cnorm[0]+=1/self.parameters[0][0]
        cnorm[1]+=1/self.parameters[0][1]
        cnorm[2]+=3/num_grid_z
        
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