#!/usr/bin/python

class TAsciiColors:
  Header  = '\033[95m'
  OKBlue  = '\033[94m'
  OKGreen = '\033[92m'
  Warning = '\033[93m'
  Fail    = '\033[91m'
  EndC    = '\033[0m'

print TAsciiColors.Header   ,'Header  Hoge hoge',TAsciiColors.EndC
print TAsciiColors.OKBlue   ,'OKBlue  Hoge hoge',TAsciiColors.EndC
print TAsciiColors.OKGreen  ,'OKGreen Hoge hoge',TAsciiColors.EndC
print TAsciiColors.Warning  ,'Warning Hoge hoge',TAsciiColors.EndC
print TAsciiColors.Fail     ,'Fail    Hoge hoge',TAsciiColors.EndC

print TAsciiColors.OKGreen  ,'Hoge hoge',TAsciiColors.Warning,'Xxx Zzz',TAsciiColors.EndC

print 'Hoge hoge'
