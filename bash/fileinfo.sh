#!/bin/bash

files=$@
if [ "$files" == "" ];then
  files="/pa.th/to/foo.tar.gz"
fi

for f in $files;do
  bn=`basename $f`
  echo "f: $f"
  echo "bn: $bn"
  echo "  basename \$f : `basename \$f`"
  echo "  \${f##*/}    : ${f##*/}"
  echo "  dirname \$f  : `dirname \$f`"
  echo "  \${f%.*}     : ${f%.*}"
  echo "  \${bn%.*}    : ${bn%.*}"
  echo "  \${f##*.}    : ${f##*.}"
  echo "  \${bn##*.}   : ${bn##*.}"
done
