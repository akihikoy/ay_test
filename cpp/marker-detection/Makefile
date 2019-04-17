##-------------------------------------------------------------------------------------------
# \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
# \version 0.1
# \date    Aug.10, 2012
##-------------------------------------------------------------------------------------------
TARGET_FLAG :=

all     : t t2 tm pt

debug	: TARGET_FLAG :=debug
debug	: all

clean	: TARGET_FLAG :=clean
clean	: all

t :
	@(set -e; make -f Makefile-t $(TARGET_FLAG))

t2 :
	@(set -e; make -f Makefile-t2 $(TARGET_FLAG))

tm :
	@(set -e; make -f Makefile-tm $(TARGET_FLAG))

pt :
	@(set -e; make -f Makefile-pt $(TARGET_FLAG))

##------------------------------------------------------------------
