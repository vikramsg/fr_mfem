import numpy as np
import re
import matplotlib.pyplot as plt


class writeGeo:

    def writeFile(self, fileName, xb, yb, xb1, yb1, xt, yt):
        op = open(fileName, 'w')

        for coun, i in enumerate(xb):
            st = 'Point('+str(coun + 1)+') = {' + str(i) + ' , ' + str(yb[coun]) + ', 0.0, 1.0 };\n'
            op.write(st)
        
        st = 'Spline(1) = {1 '
        for coun, i in enumerate(xb[1:]):
            st = st + ', ' + str(coun + 2) 
        st = st + '};'

        op.write('\n' + st + '\n\n')

        counter = len(xb) 

        for coun, i in enumerate(xb1):
            st = 'Point('+str(counter + coun + 1)+') = {' + str(i) + ' , ' + str(yb1[coun]) + ', 0.0, 1.0 };\n'
            op.write(st)
        
        st = 'Spline(2) = {' + str(counter + 1) + ' '
        for coun, i in enumerate(xb1[1:]):
            st = st + ', ' + str(counter + coun + 2) 
        st = st + '};'

        op.write('\n' + st + '\n\n')

        counter = len(xb) + len(xb1) 

        for coun, i in enumerate(xt):
            st = 'Point('+str(counter + coun + 1)+') = {' + str(i) + ' , ' + str(yt[coun]) + ', 0.0, 1.0 };\n'
            op.write(st)

        st = 'Spline(3) = {' + str(counter + 1) + ' '
        for coun, i in enumerate(xt[1:]):
            st = st + ', ' + str(counter + coun + 2) 
        st = st + '};'

        op.write('\n' + st + '\n\n')

        op.close()



class ReadP3D:

    def plot(self, xb, yb, xt, yt):

        # plot the data:
        plt.plot(xt, yt, color = "blue", linewidth =  2.5, \
                 linestyle = "dashed")

        plt.plot(xb, yb, color = "blue", linewidth =  2.5, \
                 linestyle = "dashed")

        plt.show()


    def readBlock(self, fileName):
        op = open(fileName, 'r')

        text     = op.read()

        op.close()

        textList  = text.split()

        numBlocks = int(textList[0])
        numAzim   = int(textList[1])
        numRadi   = int(textList[2])

        counter = 3

        xb  = np.zeros(numAzim)
        yb  = np.zeros(numAzim)

        xb1 = np.zeros(numAzim)
        yb1 = np.zeros(numAzim)

        xt  = np.zeros(numAzim)
        yt  = np.zeros(numAzim)

        for i in range(numAzim):
            subscript = counter + i
            xb[i]     = float(textList[subscript])
            subscript = counter + (numAzim)*(numRadi) + i
            yb[i]     = float(textList[subscript])

            subscript = counter + (numAzim)*(numRadi - 1) + i
            xt[i]     = float(textList[subscript])
            subscript = counter + (numAzim)*(numRadi + numRadi - 1) + i
            yt[i]     = float(textList[subscript])
            
            subscript = counter + (numAzim*18) + i
            xb1[i]    = float(textList[subscript])
            subscript = counter + (numAzim)*(numRadi + 18) + i
            yb1[i]    = float(textList[subscript])

#        self.plot(xb, yb, xt, yt)
        return(xb, yb, xb1, yb1, xt, yt)


class ReadTec:

    def plot(self, coords):
        numArrays = len(coords)

        for i in range(numArrays):
            if ((i > 2) and (i < 8)):
                continue
            numCoords = len(coords[i])
            xv1       = np.zeros(numCoords)
            yv1       = np.zeros(numCoords)
    
            for j in range(numCoords):
                xv1[j] = coords[i][j, 0]
                yv1[j] = coords[i][j, 1]
    
            # plot the data:
            plt.plot(xv1, yv1, color = "blue", linewidth =  2.5, \
                     linestyle = "dashed")

        plt.show()



    def read(self, fileName):
        allCoords = []

        op = open(fileName, 'r')

        lines    = op.readlines()

        numLines = len(lines)

        counter  = 0

        while (counter < numLines):
            coords, counter = self.readZone(lines, counter)

            allCoords.append(coords)

        op.close()

        return allCoords


    def readZone(self, lines, counter):
        commented = 1

        while (commented == 1):
            temp    = lines[counter] 
            counter = counter + 1
            if (temp.split()[0][0] != '#'):
                break

        getSize   = 1

        while (getSize == 1):
            if ((re.search('I', temp) is not None) and (re.search('J', temp) is not None) \
                    and (re.search('K', temp) is not None)):
                break
            temp    = lines[counter]
            counter = counter + 1
          
        numCoords = int(temp.split('=')[1].split(',')[0])

        coords = np.zeros((numCoords, 2)) 

        counter = counter + 1

        for i in range(numCoords):
            temp    = lines[counter]
            counter = counter + 1
            coords[i, 0] = float(temp.split()[0])
            coords[i, 1] = float(temp.split()[1])
        return coords, counter



if __name__=="__main__":
    run = ReadTec()

    fileName  = 'Coords.dat'

#    allCoords = run.read(fileName)
#    run.plot(allCoords)

    fileName  = 'hump_coarse.dat'
    run = ReadP3D()
    (xb, yb, xb1, yb1, xt, yt) = run.readBlock(fileName)

    fileName  = 'wall_hump.geo'
    wr = writeGeo() 
    wr.writeFile(fileName, xb, yb, xb1, yb1, xt, yt)


