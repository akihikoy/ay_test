set bashscript=$1

set tmp=/tmp/tmp$$
\bash -c "\
  printenv > $tmp-a;  \
  . $bashscript;  \
  printenv > $tmp-b"

\diff --old-group-format='' --unchanged-group-format='' $tmp-{a,b} > $tmp-c

foreach line ("`cat $tmp-c`")
  set n1 = ($line)
  set n2=`echo $n1 | sed 's/\([a-zA-Z0-9_]\+\)=\(.*\)/\1 \"\2\"/g'`
  #echo $n1
  #echo -- $n2
  eval "setenv $n2"
end

\rm $tmp-a
\rm $tmp-b
\rm $tmp-c
