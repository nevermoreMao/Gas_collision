# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 11:54:11 2018

@author: ZM1316
"""

from twodthermo import Ball
import numpy as np
import random
import matplotlib.pyplot as plt

M_B_distribution_test=False #the switch of boltzmann distribution test
pressure_test=True #the switch of pressure test

class ThreedGas():
    ''' the simulation of gas'''
    
    def __init__(self, r_ball=0.1,s_ball=50,r_container=10, num_b=50,mass_b=1e-23):
        '''initialise the a single ball in container'''
        self.__rc=r_container #radius of container
        self.__c=Ball(1e15,self.__rc,[0,0,0],[0,0,0], container=True) #container
        self.__rb=r_ball #the radius of small balls inside container
        self.__mb=mass_b # the mass of balls
        self.__q=num_b #quantity of balls
        self.__time=0 #total time elapsed
        self.__spd=s_ball#the mean speed of balls in one dimension
        self.__i=0 #the cumulative inpulse used to compute pressure
        self.__pt=0 #the time interval counter used to calculate pressure
        self.__P='TBD' #the pressure, tbd at the begining
        self.NC=0 #the number of collision used in method simulation
        self.P_data=[]# a list collect Pressure data.
                
        pos=[] # the list of positions of ball
        while len(pos) != self.__q:
            p1=[random.uniform(-(self.__rc-self.__rb),self.__rc-self.__rb),
                random.uniform(-(self.__rc-self.__rb),self.__rc-self.__rb),
                random.uniform(-(self.__rc-self.__rb),self.__rc-self.__rb)]
            
            overlap=False
            # the state if the new ball overlap with original balls
            
            for i in range(len(pos)):
                if ((p1[0]-pos[i][0])**2+(p1[1]-pos[i][1])**2+(p1[2]-pos[i][2])**2)**0.5 < 2*self.__rb+0.1:
                    #check if balls overlap with each other
                    overlap=True
                else:
                    pass
                    
            if (p1[0]**2+p1[1]**2+p1[2]**2)**0.5 >10-self.__rb:
                #check if the ball overlap with container
                pass
                #if yes, pass the ball without appending
                
            elif overlap :
                pass
            else:
                pos.append(p1)
                
        self.__pos=np.array(pos)                    
        
        vel=np.array([random.uniform(-2*self.__spd,2*self.__spd
                                     ) for i in range(3*self.__q)]).reshape(self.__q,3)
        vel[:,0] -= vel[:,0].mean()
        vel[:,1] -= vel[:,1].mean()
        vel[:,2] -= vel[:,2].mean()
        
        self.__vel=vel #velocity of balls

        self.__bs=[self.__c] # the balls
        for i in range(self.__q):
            self.__bs.append(Ball(self.__mb,self.__rb,self.__pos[i],self.__vel[i]))  
            
    def r_container(self):
        '''display the radius of container'''
        return self.__rc
    
    def balls(self):
        '''display the balls'''
        return self.__bs
    
    def energy(self):
        '''compute the kinetic energy of the current state'''
        ke_array=np.array(list(0.5*b.mass()*np.dot(b.vel(),b.vel()) for b in self.balls()[1:]))
        energy=ke_array.sum()
        return energy
    
    def momentum(self):
        ''' compute the total momentum of the current state'''
        p_array=np.array(list( b.mass()*b.vel() for b in self.balls()[1:]))
        p=p_array.sum()
        if p < 1e-25:
            p=0
        return p       
    
    def var(self):
        '''compute the variance of speed of balls'''
        bs_spd=np.array(list(b.speed() for b in self.balls()[1:]))
        return bs_spd.var()
    
    def temp(self):
        '''comput the temperature of current state'''
        mean_ke=self.energy() /self.__q
        k=1.38064852e-23
        temp=2*mean_ke/(3*k)
        return temp
    
    def cmmt(self):
        '''compute of the change in momentum of the ball when collide with wall'''
        vmmt=self.mass() * self.vel()#momentum vector of ball
        cmmt=2*np.dot(vmmt,self.pos())/(np.sqrt(np.dot(self.pos(),self.pos())))
        #change in momentum
        mmt=np.sqrt(np.dot(cmmt,cmmt))
        return mmt
        
    def time(self):
        '''display the time elapsed'''
        return self.__time
    
    def simulation(self):
        '''a very simple simulation of the motion of balls without animation'''
        p_tc=[] #all calculated time to collide between two single balls
        bs_c=self.balls()[:] #a copy of the balls, the ball to collide
               
        for i in range(self.__q):
            b1=bs_c.pop(0)
            
            for j in range(len(bs_c)):
                dt=b1.time_to_collision(bs_c[j])
                p_tc.append((i,j+i+1,dt))

        c_set=min([i for i in p_tc] ,key=lambda x: x[2])
        #the tuple of collide set[0] and set[1] are the next two balls to 
        #collide, set[2]is next time to collision
        
        tc=c_set[2]
        
        for b in self.__bs:
            b.move(tc)
        
        if self.__bs[c_set[0]].container():
            I=self.__bs[c_set[1]].cmmt()
            self.__i += I
            #record cumulative inpulse
                        
        elif self.__bs[c_set[1]].container():
            I=self.__bs[c_set[0]].cmmt()
            self.__i += I
            
        else:
            pass
        
        self.__bs[c_set[0]].collide(self.__bs[c_set[1]])
                
        self.NC +=1
        self.__time += tc #time elapsed
        self.__pt += tc #time counter used to find pressure
        
        if self.__pt >= 0.5:
            #compute pressure over time interval self.__pt
            self.__P=self.__i/self.__pt/(4*np.pi*self.r_container()**2)
            self.P_data.append(self.__P)

            self.__pt=0 #reset the time counter pt
            self.__i=0#reset inpulse counter i

#-----------------obtain the data-------
if M_B_distribution_test ==True:
    mb_test=ThreedGas(r_ball=0.01,s_ball=20,mass_b=1e-23,num_b=3000)
                      
    while mb_test.NC != 50000:
        mb_test.simulation()
        print(mb_test.NC)
        print(mb_test.temp())

    fig3=plt.figure(3)
    ax3=plt.axes()
    ax3.axes.set_aspect('equal')        
    
    v=list(b.speed() for b in mb_test.balls()[1:])

    with open('3ddata.txt','w') as f:
        for i in v:
            f.write(str(i)+'\n')
         
    plt.hist(v,50, facecolor='green',alpha=0.75)        
    plt.show()        
    print(f'variance={mb_test.var()}')
    print(f'T={mb_test.temp()}K')
    
if pressure_test==True:
    
    
    p_gas=ThreedGas(s_ball=50)
    while len(p_gas.P_data) !=5:
        #take five pressure and take the mean
        p_gas.simulation()
    print(f'mean preesure={np.array(p_gas.P_data).mean()}')
    print(f'std={np.array(p_gas.P_data).std()}')    
    print(f'temperature={p_gas.temp()}')
    
    print(f'{p_gas.temp()}'+'\t'+f'{np.array(p_gas.P_data).mean()}'+
             '\t'+f'{np.array(p_gas.P_data).std()}')
