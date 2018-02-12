
import os

# Check version
#  Python 2.7.12 on win32 (Windows version)


IN_DIR = 'wav'
OUT_DIR = 'spectrogram'


if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

from getspecgram1 import GetSpecgram
GetSpecgram1= GetSpecgram()

from  conv2image import *

# number letter
import re
pattern=r'([0-9]+)'

loop=0
xspec = np.array( [] )

mode=1  # 0: max, min search  1: image write out

# scan and copy to OUT_DIR via head letter
for f in (os.listdir(IN_DIR)):
    # check head letter
    if f[0:1].isdigit() :
        source = os.path.join(IN_DIR, f)
        bname, ext = os.path.splitext( f )
        
        # reject some input wav file
        # pick up number, second item
        # reject area is 
        #   xxx40 <=     xxx260 >= 
        a=re.findall(pattern,bname)
        b=map(int,a)
        #print (b)
        if b[1] <= 40 :
        	continue
        elif b[1] >= 260:
        	continue
        
        f2= bname + '.' + 'png'
        dest = os.path.join(OUT_DIR, f2)
        #print ( source, dest )
        if mode == 1:
        	spec=GetSpecgram1.get(source)
        	conv2image(spec, dest, Fshow=False)
        else:
        	spec=GetSpecgram1.get(source)
        	print ("filename, max,min",f,  np.max(spec), np.min(spec) )
        	xspec=np.append(xspec, np.max(spec))
        	xspec=np.append(xspec, np.min(spec))
        loop+=1
        
#    if loop >= 2850:
#        break

if mode == 1:
     pass
else:
     print ("loop, max,min", loop, np.max(xspec), np.min(xspec) )

### ('loop, max,min', 2850, 44.381695791912605, -138.56194539669562)
