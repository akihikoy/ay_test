program func1
  implicit none
  real :: a=3, b=4, c=5, S, area
  S= area(a,b,c)
  print *,"a=",a,"b=",b,"c=",c
  print *,"area=",S
end program func1

function area(a,b,c)
  implicit none
  real :: a,b,c,p,area
  p= (a+b+c)/2.0
  area= sqrt(p*(p-a)*(p-b)*(p-c))
end function area

