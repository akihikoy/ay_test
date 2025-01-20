#!/usr/bin/python3
#\file    palindromes.py
#\brief  Code for: https://codefights.com/challenge/ddNHWesRJHpSv2xWG/main?utm_source=facebook&utm_medium=cpc&utm_campaign=Solve_A_Challenge_V2
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.10, 2016

#Generate all different strings whose contents are defined by table.
def GetAllCombinations(table):
  strings= []
  for c,s in table.items():
    if s==0:  continue
    table[c]-= 1
    substrings= GetAllCombinations(table)
    if len(substrings)==0:
      substrings= [c]
    else:
      substrings= [c+string for string in substrings]
    table[c]+= 1
    strings+= substrings
  return strings

#Return all palindromes of s.
def Palindromes(s):
  table={}  #Map charactor:count in s
  for c in s:
    if c in table:  table[c]+= 1
    else: table[c]= 1
  odd= [c for n in table.values() if n%2==1]
  if len(odd)>1:  return []  #Impossible to generate palindrome for e.g. 'aaabbb'
  center= ''
  if len(odd)==1:
    center= odd[0]
    table[center]-= 1
  half_table= {c:n/2 for c,n in table.items()}
  half_strings= GetAllCombinations(half_table)
  if len(half_strings)==0:
    palindromes= [center] if center!='' else []
  else:
    palindromes= [string+center+string[::-1] for string in half_strings]
  return palindromes

if __name__=='__main__':
  TEST= ['abc','aabbc','ababb','a','ab','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa','zzzz','aabbbbaa']
  for s in TEST:
    palindromes= Palindromes(s)
    print(s,': found(%d) :'%len(palindromes),palindromes)
