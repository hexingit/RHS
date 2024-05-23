# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:39:52 2024 
@author: hex ：phased array + hologaphic BF in satllete; 
"""
# # ---- Learn to  Optimize 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np 
import matplotlib.pyplot as plt
import torch as t 
from math import e
import math
import pickle 
import time 
 
def GP(H ,E, L, hypm ,Pt):
    # ------- Projection Gradient Ascent execution --------- 
    Tno=H.size()[0];  Nr=H.size()[1];  Nt=H.size()[2] ;  
    # ---- initializing variables;  
    w = t.randn(Tno, Nr,L,dtype=t.cfloat) ;w = w  /t.abs(w)  
    m = t.ones(Tno, Nt,1 )#small matrix to control L feeds
    M=m *E 
    gradw2=t.ones(Tno, Nr,L,dtype=t.cfloat) 
    A=t.ones(Tno, 1,Nt,L,dtype=t.cfloat);   
    obj0=0     
    while 1:   
        # # grad ascent for Amptitude m
        gradm=t.zeros(Tno, Nt,1,dtype=t.cfloat)
        for l in range(L):   
            wt=w[:,:,l].reshape(( Tno, Nr,1))
            A[:,:,:,l]= (t.transpose(M,1,2).conj()@t.transpose(H[:,:,:,l],1,2).conj()@wt@t.transpose(wt,1,2).conj()@H[:,:,:,l]) 
            gradm += t.transpose(A[:,:,:,l],1,2)*E /(t.abs(t.transpose(wt,1,2).conj()@H[:,:,:,l]@M)**2+Nr*1 )  
        m=m+  hypm * gradm.real 
        m[m>=1]=1; m[m<=0]=0#projection continous 
        M=m *E
        M = t.sqrt(Pt / ( t.linalg.matrix_norm(M,ord='fro')**2)).reshape(Tno,1,1) *M #power normalization

        # # grad ascent for analog Rec w
        for l in range(L):  
            gradw2[:,:,l]= (H[:,:,:,l]@M).reshape(( Tno, Nr)) 
        w=gradw2  # close-form solution
        w = w  /t.abs(w)
        
        obj1=sum_capcity( w, H,M, Tno) 
        if abs(obj1-obj0)/obj1<=1e-4:
            break
        obj0=obj1
    return   obj1,w,M 

def phaseArray(  H , L, hypm  ,Pt):
    # ------- Projection Gradient Ascent execution --------- 
    Tno=H.size()[0];  Nr=H.size()[1];  Nt=H.size()[2] ; 
    # ---- initializing variables;  
    w = t.randn(Tno, Nr,L,dtype=t.cfloat) ;    
    M=t.randn(Tno, Nt,1,dtype=t.cfloat) ;m = M  ;  
    gradw2=t.ones(Tno, Nr,L,dtype=t.cfloat) 
    A=t.ones(Tno, Nt,1,L,dtype=t.cfloat);   
    obj0=0    ;  scaler=(Pt/Nt)**(1/2)
    while 1:   
        # # grad ascent for Amptitude m
        gradm=t.zeros(Tno, Nt,1,dtype=t.cfloat)
        for l in range(L):   
            wt=w[:,:,l].reshape(( Tno, Nr,1))
            A[:,:,:,l]= t.transpose(H[:,:,:,l],1,2).conj()@wt@t.transpose(wt,1,2).conj()@H[:,:,:,l]@M 
            gradm +=  A[:,:,:,l] /(t.abs(t.transpose(wt,1,2).conj()@H[:,:,:,l]@M)**2+Nr*1 )  
        m=m+  hypm * gradm 
        m = m  /t.abs(m)
        M =   scaler*m #power normalization

        # # grad ascent for analog Rec w
        for l in range(L):  
            gradw2[:,:,l]= (H[:,:,:,l]@M).reshape(( Tno, Nr))  
        w = gradw2  /t.abs(gradw2) 
        
        obj1=sum_capcity( w, H,M, Tno) 
        if abs(obj1-obj0)/obj1<=1e-4:
            break
        obj0=obj1
    return   obj1,w,M 
def sum_capcity( w, H,M, batch_size):
    Nr=w.size()[1]; L=w.size()[2]
    obj=t.zeros(  batch_size,1,1 );  
    for l in range(L): 
        wt=w[:,:,l].reshape(( batch_size, Nr,1))
        obj+= t.log2( 1+t.square(t.abs(t.transpose(wt,1,2).conj()@ H[:,:,:,l]@M ))/Nr ) 
    return   sum(obj ) / batch_size
def unrollGP(   H ,E, L, hypm ,Pt):
    # ------- Projection Gradient Ascent execution --------- 
    Tno=H.size()[0];  Nr=H.size()[1];  Nt=H.size()[2] ; lop1=len(hypm)
    # ---- initializing variables;  
    w = t.randn(Tno, Nr,L,dtype=t.cfloat) ;w = w  /t.abs(w)  
    m = t.ones(Tno, Nt,1 )#small matrix to control L feeds
    M=m *E 
    gradw2=t.ones(Tno, Nr,L,dtype=t.cfloat) 
    A=t.ones(Tno, 1,Nt,L,dtype=t.cfloat);   
    for i in range(lop1):  
        # # grad ascent for Amptitude m
        gradm=t.zeros(Tno, Nt,1,dtype=t.cfloat)
        for l in range(L):   
            wt=w[:,:,l].reshape(( Tno, Nr,1))
            A[:,:,:,l]= (t.transpose(M,1,2).conj()@t.transpose(H[:,:,:,l],1,2).conj()@wt@t.transpose(wt,1,2).conj()@H[:,:,:,l]) 
            gradm += t.transpose(A[:,:,:,l],1,2)*E /(t.abs(t.transpose(wt,1,2).conj()@H[:,:,:,l]@M)**2+Nr*1 )  
        m=m+  hypm[i]* gradm.real 
        m[m>=1]=1; m[m<=0]=0#projection continous 
        M=m *E
        M = t.sqrt(Pt / ( t.linalg.matrix_norm(M,ord='fro')**2)).reshape(Tno,1,1) *M #power normalization

        # # grad ascent for analog Rec w
        for l in range(L):  
            gradw2[:,:,l]= (H[:,:,:,l]@M).reshape(( Tno, Nr)) 
        w=gradw2  # close-form solution
        w = w  /t.abs(w) 
    return   w,M    
def genE( lamd,de,Nx,L):
    x= t.arange(Nx) *de #corodinate of element
    y=x
    xf=x[0:L]+de/2 #corodinate of feeds
    yf=Nx*de/2*t.ones(L)
    E=t.ones(Nx*Nx,L,dtype=t.cfloat)
    for i in range(Nx):
        for j1 in range(Nx):
            for k in range(L):
                E[i*Nx+j1,k]=e**(-1j*2*np.pi* math.sqrt(3)/lamd* math.sqrt((x[i]-xf[k])**2+(y[j1]-yf[k])**2) ) 
    return  E
 
def Hsat2(lamd,Nx,d0,de,thetaMax,Nr,Nt,train_size):#de is the distance between elements
    H  = t.zeros(train_size,Nr,Nt,L,dtype=t.cfloat);
    nx=t.arange(Nx).reshape((Nx,1)).repeat(Nx,1)
    for l in range(L):
        for t0 in range(train_size):
            theta=t.rand(1)*thetaMax
            for nr in range(Nr):  
                        dx=d0*t.tan(theta)+nr*lamd/2-(nx*de-Nx/2*de) 
                        dz=d0
                        H[t0,nr,:,l]=(lamd/(4*np.pi*(dx**2+dz**2)**0.5)*t.exp(-1j*2*np.pi/lamd*(t.sin(theta)*dx+t.cos(theta)*dz))).reshape(1,1,Nt)
    return H
# ---- ------------------------main -------------------------- 
dmode= 1 #1=change RHS size; 2= change tranmit power 
Nr =   2     # Rec antennas
L= 4  #user number=RF chain number
lamd=1e-2; #30GHz
scale=1e9
SNR=  200 ;
Pt=10**(SNR/10)*(1/scale)**2
d0=100*1e3#altitude of sat 100km
thetaMax=np.pi/6 # random max angle
deH=lamd/5  #de is the distance between elements
 
test_size = 100   
RFno=1;#feed number
 
# ----------------RHS: classical GP----------------------------- 
stepSize1= 1e1;   
#------------------Phase array-------------------------------
deP=lamd/2  #de is the distance between elements
lamdAmp=deP/deH  ;  
# ------------------RHS: Unfolded GP -------------------------- 
file = open(f'HoLoNx30Lop5.pkl', 'rb')  
mu2m= pickle.load(file); file.close();   
if dmode==1:
    lopPara=[10,20,30,40] #Nx parameter in a loop
    # lopPara=[40,60,80,100] #Nx parameter in a loop
elif dmode==2:
    lopPara=[2,4,6,8,10] #transmit power (watt) in a loop

RateAll=t.zeros([3,len(lopPara)]); TimeAll=np.zeros((3,len(lopPara)));IterAll=len(mu2m[0])*np.ones((3,len(lopPara)));
with t.no_grad():
    for lop in range(len(lopPara)):
        print(lop/len(lopPara))
        if dmode==1:
            Nx = int(lopPara[lop] )     # Num of users 
            Nt =   Nx**2    # Tx antennas
        elif dmode==2:
           SNR=  200 -10*np.log10(10/lopPara[lop]);
           Pt=10**(SNR/10)*(1/scale)**2 
           Nx=30; Nt =   Nx**2    # Tx antennas
        H_testD =  Hsat2(lamd,Nx,d0,deH,thetaMax,Nr,Nt,test_size)*scale
        E0=genE(lamd,deH,Nx,RFno).reshape(1,Nt,RFno); E=E0.repeat(test_size,1,1)
        Nxp=math.floor(Nx/lamdAmp) #phase array number in x-axis
        H_testPhase =  Hsat2(lamd ,Nxp,d0,deP,thetaMax,Nr,Nxp**2,test_size)*scale  
 
        # ---------------- -----GP on the test set------------------- 
        time_start = time.time() # 
        RateAll[0,lop],w,M =  GP(H_testD,E, L,stepSize1,Pt) 
        time_end = time.time() ;TimeAll[0,lop]= time_end - time_start  ;#print('GP time cost', TimeAll[0,lop], 's') #运行所花时间 
        # --------------------- unrolled GP ------------------------ 
        time_start = time.time() # 
        w,M= unrollGP(H_testD,E,L,mu2m[0],Pt) 
        time_end = time.time() ;TimeAll[1,lop]= time_end - time_start  ;#print('Unrolled GP time cost', TimeAll[1,lop], 's') #运行所花时间 
        RateAll[1,lop]=sum_capcity(w,H_testD,M, test_size)
        # -----------------------Phase array--------------------------
        time_start2 = time.time() # 
        RateAll[2,lop],w,M =phaseArray(H_testPhase, L, stepSize1  ,Pt) 
        time_end2 = time.time() ;TimeAll[2,lop]= time_end2 - time_start2  ;#print('Phased array time cost', TimeAll[2,lop], 's') #运行所花时间 

# ploting the results
plt.figure()  
plt.plot(lopPara, RateAll[1].detach().numpy(), 'b-o',label='RHS (Unrolled GP)') 
plt.plot(lopPara, RateAll[0].detach().numpy(), 'r-*', label='RHS (GP)') 
plt.plot(lopPara, RateAll[2].detach().numpy(), '--',label='Phased array') 
if dmode==1:
    plt.xlabel('Nx: Number of RHS elements in x-axis (Nx=Ny)')
elif dmode==2:
    plt.xlabel('Transmit power (W)')
plt.ylabel('Sum Rate (bps/Hz)')
plt.legend(loc='best')
plt.grid()
# plt.savefig(f'Figs/capacityD{dmode}SNR{SNR}.eps', bbox_inches='tight')
plt.show() 

plt.figure()
plt.semilogy(lopPara, TimeAll[2]/test_size , '--', label='Phased array')   
plt.semilogy(lopPara, TimeAll[0]/test_size, 'r-*', label='RHS (GP)') 
plt.semilogy(lopPara, TimeAll[1]/test_size, 'b-o',label='RHS (Unrolled GP)') 
if dmode==1:
    plt.xlabel('Nx: Number of RHS elements in x-axis (Nx=Ny)')
elif dmode==2:
    plt.xlabel('Transmit power (W)')
plt.ylabel('Compution Time (s)')
plt.legend(loc='best')
plt.grid()
# plt.savefig(f'Figs/TimeD{dmode}SNR{SNR}.eps', bbox_inches='tight')
plt.show() 

 
  
# Saving the objects:
# file = open(f'./Figs/TimeGMode{dmode}Test{test_size}.pkl', 'wb') 
# pickle.dump(  [RateAll,TimeAll,lopPara,dmode]  , file);file.close() 