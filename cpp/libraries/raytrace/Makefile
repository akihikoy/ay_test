##-------------------------------------------------------------------------------------------
# \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
# \version 0.1
# \date    Feb.02, 2015
##-------------------------------------------------------------------------------------------
TARGET_FLAG :=

all     : lib exe

debug	: TARGET_FLAG :=debug
debug	: all

clean	: TARGET_FLAG :=clean
clean	: all

lib :
	@(set -e; make -f Makefile.doncross $(TARGET_FLAG))

exe :
# 	@(set -e; make -f Makefile.exe $(TARGET_FLAG))

##------------------------------------------------------------------
