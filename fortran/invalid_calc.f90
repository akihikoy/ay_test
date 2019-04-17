program invalid_calc
implicit none
real :: zero=0.0, neg1=-1.0, onedotone=1.1
print *,"Zero division 1.0/0.0 = ", 1.0/zero
print *,"sqrt(-1.0) = ",sqrt(neg1) 
print *,"asin(1.1) = ",asin(onedotone)
!rad=0.1+0.1+0.1+0.1+0.1+0.1+0.1+0.1+0.1
print *,"0.1=",0.1
print *,"0.1+0.1+0.1+0.1+0.1+0.1+0.1+0.1+0.1=",0.1+0.1+0.1+0.1+0.1+0.1+0.1+0.1+0.1
print *,"0.1*9=",0.1*9
print *,"0.9=",0.9
print *,"(0.1*9==0.9) = ",(0.1*9==0.9)
print *,"0.9/9=",0.9/9
print *,"(0.9/9)*9=",(0.9/9)*9
print *,"((0.9/9)*9==0.9) = ",((0.9/9)*9==0.9)

!do rad=0.1,2.0,0.1
!  print *,rad,(rad/70)*70 - rad
!enddo

print *,""
stop
end program invalid_calc

