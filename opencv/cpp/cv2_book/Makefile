##------------------------------------------------------------------------------
#     Copyright (C) 2011  Akihiko Yamaguchi
##------------------------------------------------------------------------------
EXEC = optflow-gmmem.out
LIBOBJS =
CXX = g++
AR = ar
CXXFLAGS := -g -Wall -rdynamic -O2
LDFLAGS  := $(CXXFLAGS)
RMFLAGS  :=  -vf
##------------------------------------------------------------------------------
## make  *.out  from  *.c and *.cpp
OBJS := \
      $(patsubst %.c,%.o,$(wildcard *.c)) \
      $(patsubst %.cpp,%.o,$(wildcard *.cpp)) #
##------------------------------------------------------------------------------
LIBS    = -I/usr/include -I/usr/local/include
LDLIBS  = -lm -L/usr/local/lib
##------------------------------------------------------------------------------
## for OpenCV
LIBS   += `pkg-config opencv --cflags`
LDLIBS += `pkg-config opencv --libs`
##------------------------------------------------------------------------------
all : $(EXEC)
##------------------------------------------------------------------------------
## generic make rules

clean   :
	@rm $(RMFLAGS) *.o *.a *.il $(EXECS) $(ARC) $(EXTRA_CLEAN)
	@find -name '*.d' -type f | xargs rm $(RMFLAGS)

$(EXEC) : $(OBJS) $(LIBOBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS) $(LIBOBJS) $(LDLIBS)
	@(echo "$(EXEC) is generated")

%.o : %.cpp
	$(CXX) $(LIBS) $(CXXFLAGS) -o $@ -c $<

DEPOPT  ?= -M
# DEPOPT  ?= -MM

%.d: %.cpp
	@(rm -f $@; \
	$(CXX) $(DEPOPT) $(LIBS) $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	[ -s $@ ] || rm -f $@; \
	rm -f $@.$$$$; \
	)
##------------------------------------------------------------------------------
include $(OBJ:.o=.d) $(OBJS:.o=.d) $(objects:.o=.d) $(addsuffix .d, $(BINARY))
##------------------------------------------------------------------------------
