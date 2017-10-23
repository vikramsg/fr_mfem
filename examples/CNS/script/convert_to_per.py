import numpy as np
import re


class ConvertPer:
    '''
    We input MFEM meshes, specify which boundary tags are periodic
    and convert two boundaries into zero boundaries by modiying
    topology

    Right now this works with perioidicity in only one direction
    And only for quads

    The order of operations for a GMSH file is to specify the two
    periodic boundaries, generate the mesh and then input into 
    MFEM and output in MFEM format which is input here
    '''

    def __init__(self):
        '''
        Specify tags of the two boundaries that are periodic to each other
        '''
        self.periodic_tags = [1, 2] # Tags of boundaries that are periodic
        self.eps           = np.finfo(float).eps #Machine tolerance

    def convert(self, fileName, newFileName):
        ndim, vert = self.getVertices(fileName) # Get all vertices

        bds        = self.getBoundaries(fileName, ndim) #Get boundary connectivity
        
        self.modVertices(ndim, vert, bds, fileName, new_fileName)


    def writeVertices(self, ndim, vert, fileName):
        '''
        Write vertices in the GEOMETRY format of MFEM v1.0
        They are specified for each element with increasing x for each y
        '''
        fop = open(fileName, 'r')

        lns = fop.readlines()

        for it, i in enumerate(lns):
            if re.search('elements', i) is not None:
                break
        cn = it + 1
        num_ele = int(lns[cn])
        cn = cn + 1

        st = 'nodes\nFiniteElementSpace\nFiniteElementCollection: L2_T1_2D_P1\n'
        st = st + 'VDim: 2\nOrdering: 1\n \n'

        coun = 0
        for it, i in enumerate(lns[cn:]):
            if len(i.strip()) == 0: #Skip empty lines
                continue
            ln  = i.split()

            tag = int(ln[1])

            eleArray = np.zeros((len(ln) - 2, ndim))
            for jt, j in enumerate(ln[2:]):
                eleArray[jt] = vert[int(j)]

            if tag == 3:
                newArray  = np.zeros((len(ln) - 2, ndim))
                for jt, j in enumerate(eleArray):
                    newArray[jt] = eleArray[jt]
                newArray[3] = eleArray[2]
                newArray[2] = eleArray[3]

                for jt, j in enumerate(newArray):
                    lnst = ''    
                    for kt, k in enumerate(j):
                        lnst = lnst + str(k) + '\t'
                    st = st + lnst.rstrip() + '\n'
                st = st + '\n'
    
                coun = coun + 1
                if coun == num_ele:
                    break

        st = st.rstrip() #Remove trailing whitespace

        fop.close()

        return st



    def modVertices(self, ndim, vert, bds, fileName, new_fileName):
        nvert   = vert.shape[0]
        nbds    = bds.shape[0]

        bd_vert = self.getPerBndVert(ndim, bds)
        assoc   = self.getAssoc(ndim, bd_vert, vert) 

        fop = open(fileName, 'r')
        nop = open(new_fileName, 'w')

        lns = fop.readlines()

        for it, i in enumerate(lns):
            nop.write(i)
            if re.search('elements', i) is not None:
                break

        cn = it + 1
        num_ele = int(lns[cn])
        nop.write(lns[cn])
        cn = cn + 1
 
        coun = 0
        for it, i in enumerate(lns[cn:]):
            if len(i.strip()) == 0: #Skip empty lines
                continue
            ln = i.split()
            st = ''

            st = st + ln[0] + '\t' + ln[1] + '\t'

            for j in ln[2:]:
                notAssoc = True
                assNode  = None 
                for kt, k in enumerate(assoc):
                    if (int(j) == k[1]):
                        notAssoc = False
                        assNode  = k[0]
                if notAssoc == True:
                    st = st + j + '\t'
                else:
                    st = st + str(assNode) + '\t'
            st = st.rstrip()

            st = st + '\n'
        
            nop.write(st)

            cn = cn + 1

            coun = coun + 1
            if coun == num_ele:
                break


        for it, i in enumerate(lns[cn:]):
            nop.write(i)
            cn = cn + 1
            if re.search('vertices', i) is not None:
                break
            
        nop.write(lns[cn] + '\n')

        st = self.writeVertices(ndim, vert, fileName)

        nop.write(st)


        fop.close()
        nop.close()



    def getPerBndVert(self, ndim, bds):
        '''
        Get boundary vertices that have periodic tags
        '''
        bd_vert = [[], []]

        lo = min(self.periodic_tags)
        hi = max(self.periodic_tags)

        cn_lo = 0
        cn_hi = 0
        for it, i in enumerate(bds):
            tag  = i[0]

            per_bnd = False
        
            for j in self.periodic_tags:
                if tag == j:
                    per_bnd = True
                    break

            ver   = np.zeros((ndim      ), dtype = int) 
            if tag == lo and per_bnd == True:
                bd_vert[0].append([])
                for jt, j in enumerate(i[2:]):
                    ver[jt] = j
                bd_vert[0][cn_lo] = ver
                cn_lo = cn_lo + 1
            elif tag == hi and per_bnd == True:
                bd_vert[1].append([])
                for jt, j in enumerate(i[2:]):
                    ver[jt] = j
                bd_vert[1][cn_hi] = ver
                cn_hi = cn_hi + 1

        bd_vert = np.array(bd_vert)

        return bd_vert
 
    def getAssoc(self, ndim, bd_vert, vert):
        '''
        Get the periodic vertex associations 
        '''
        assoc   = [] 

        for i in bd_vert[0]:
            for j in i:
                for p in bd_vert[1]:
                    for q in p:
                        dist = self.vertDist(ndim, vert[j], vert[q])
                        for m in dist:
                            if m < self.eps:
                                assoc.append([j, q])
        
        assoc = np.array(assoc)

        assoc = np.vstack({tuple(row) for row in assoc}) # Get unique rows

        return assoc


    def vertDist(self, ndim, vert1, vert2):
        dist = np.zeros(ndim)
        for i in range(ndim):
            dist[i] = abs(vert1[i] - vert2[i])

        return dist


    def getBoundaries(self, fileName, ndim):
        op  = open(fileName, 'r')
        lns = op.readlines()
        op.close()

        for it, i in enumerate(lns):
            if re.search('boundary', i) is not None:
                break

        cn = it + 1
 
        for it, i in enumerate(lns[cn:]):
            if (len(lns[it].strip())) > 0:
                break

        cn      = cn + it 
        num_bds = int(lns[cn])

        cn      = cn + 1 

        bds     = np.zeros((num_bds, ndim + 2), dtype = int) # This works only for 2D
                                                             # Since it assumes only 2 points in a boundary
        coun    = 0
        for it, i in enumerate(lns[cn:]):
            if len(i.strip()) == 0:
                continue
            ln = i.split()
            for jt, j in enumerate(ln):
                bds[coun, jt] = j

            coun = coun + 1

            if (coun == num_bds):
                break

        return(bds)




    def getVertices(self, fileName):
        op  = open(fileName, 'r')
        lns = op.readlines()
        op.close()

        for it, i in enumerate(lns):
            if re.search('vertices', i) is not None:
                break

        cn = it + 1
            
        for it, i in enumerate(lns[cn:]):
            if (len(lns[it].strip())) > 0:
                break

        cn = cn + it 
        num_vertices = int(lns[cn])

        cn = cn + 1

        for it, i in enumerate(lns[cn:]):
            if len(i.strip()) > 0:
                break

        cn   = cn + it 
        ndim = int(lns[cn])

        cn   = cn + 1

        vert = np.zeros((num_vertices, ndim))
        coun = 0
        for it, i in enumerate(lns[cn:]):
            if len(i.strip()) == 0:
                continue
            ln = i.split()
            for jt, j in enumerate(ln):
                vert[coun, jt] = j

            coun = coun + 1

        return ndim, vert


if __name__=="__main__":
    fileName     = 'sq_mfem.mesh'
    new_fileName = 'per_sq_mfem.mesh'

    rn = ConvertPer()
    rn.convert(fileName, new_fileName)
