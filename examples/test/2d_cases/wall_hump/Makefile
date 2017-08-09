EXE = test1.x

CXX =  mpicxx -g -O3 -I../../../../ -I/home/vikram/usr/local/metis/include -I/home/vikram/usr/local/hypre/include 

LDFLAGS = -L../../../../ -lmfem -L/home/vikram/usr/local/metis/lib -lmetis -L/home/vikram/usr/local/hypre/lib -lHYPRE -lrt

all:  $(EXE)

test1.x: t1.cpp 
	$(CXX) $^  $(LDFLAGS) -o $@ 

9ex.x: ex9.cpp 
	$(CXX) $^  $(LDFLAGS) -o $@ 


clean: 
	rm *.x 

