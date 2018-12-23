# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 21:00:41 2018

@author: MZS
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

M_B_distribution_test=False #switch of Maxwell Boltzmann distribution test
animate_2d=True #switch of animation of 2d snookers in container
histogram_2d=False #switch of animation of living histogram
ifbrow=False #switch of brownian motion
ifdiatm=False #switch of adding second sort of balls to 2d aniamtion

class Ball:
    '''
    Simple ball class.
    Uses a matrix representation to move the ball in circle
    '''
    
    def __init__(self,mass,radius,position,velocity,clr='r', container=False):
        '''
        initialise the ball
        postion and velocity are list or array
        '''        
        self.__mass=float(mass) #mass
        self.__radius=float(radius) #radius
        #Force the mass and radius to be float even the input is int
        self.__container = container #if the ball is a container
        self.__pos=np.array(position,dtype=float)
        self.__vel=np.array(velocity,dtype=float)
        #Make the position and velocity numpy array

        if self.__container == True:
            self.__patch=plt.Circle(self.__pos,0, fc=clr)
            #radius of patch is 0, so dont show the container in patch
        else:
            self.__patch=plt.Circle(self.__pos,self.__radius, fc=clr,fill=True)
        #Use a patch to animate the ball
        
    def pos(self):
        '''display the position of the ball'''
        return self.__pos
    
    def vel(self):
        '''display the velocity of the ball'''
        return self.__vel
    
    def speed(self):
        '''display the speed of the ball'''
        return np.sqrt(np.dot(self.vel(),self.vel()))
    
    def mass(self):
        '''display the mass of the ball'''
        return self.__mass
    
    def container(self):
        '''check if the ball is a container'''
        return self.__container
    
    def cmmt(self):
        '''compute of the change in momentum of the ball when collide with wall'''
        vmmt=self.mass() * self.vel()#momentum vector of ball
        vcmmt=2*np.dot(vmmt,self.pos())/(np.sqrt(np.dot(self.pos(),self.pos())))
        #change in momentum
        cmmt=np.sqrt(np.dot(vcmmt,vcmmt))
        return cmmt
        
    def move(self,dt):
        '''move the ball to a new position and move the patch too'''
        self.__pos += (dt*self.__vel)
        self.__patch.center=self.__pos
        return self
    
    def time_to_collision(self,other):
        '''calculate the time until the next collision between
        this ball and another one, or the container.'''
        dr=self.pos()-other.pos()
        dv=self.vel()-other.vel()
        
        a=np.dot(dv,dv) #the parameter of t**2
        b=2*np.dot(dr,dv) #the parameter of t
        
        if a == 0:
            dt=np.inf
        
        else:            
            if self.__container or other.__container:
                #The collision between ball and container
                c=np.dot(dr,dr)- (self.radius()-other.radius())**2
            else:
                c=np.dot(dr,dr)- (self.radius()+other.radius())**2
                
            if b**2-4*a*c <= 1e-10:
                dt= np.inf
            
            else:
                dt1= (-b+ (b**2-4*a*c)**0.5) / (2*a)
                dt2= (-b- (b**2-4*a*c)**0.5) / (2*a)
                soln=[dt1,dt2]            
                if max(soln) <= 1e-15:
                    #use small number rather than 0 to avoild rounding error
                    dt=np.inf
                else:
                    dt=min(i for i in soln if i >1e-15)
        return dt
        
    def collide(self,other):
        '''
        simulate the collision between balls(self and ball)
        change the velocity after collision.
        '''
        
        v1i=self.vel() #initial speed of self
        v2i=other.vel() #initial speed of other
        m1=self.__mass#mass of self        
        m2=other.__mass#mass of other ball
        x1=self.pos() #position of self
        x2=other.pos() #position of other ball
        
        if self.__container==True:
            #if self is container
            v1f=0
            v2f=v2i-2*np.dot((v2i-v1i),(x2-x1))/np.dot((x2-x1),(x2-x1))*(x2-x1)
            
        elif other.__container==True:
            #if other is container
            v1f=v1i-2*np.dot((v1i-v2i),(x1-x2))/np.dot((x1-x2),(x1-x2))*(x1-x2)
            v2f=0
        
        else:     
            #if two ball collide
            v1f=v1i-2*m2/(m1+m2)*np.dot((v1i-v2i),(x1-x2))/np.dot((x1-x2),
                          (x1-x2))*(x1-x2)
            
            v2f=v2i-2*m1/(m2+m1)*np.dot((v2i-v1i),(x2-x1))/np.dot((x2-x1),
                          (x2-x1))*(x2-x1)
            
        self.__vel=v1f #change the velocity of self
        other.__vel=v2f #change the velocity of other
        return self,other
    
    def get_patch(self):
        '''display the patch representing the ball'''
        return self.__patch
    
    def radius(self):
        '''display the radius'''
        return self.__radius    
    
class Gas:
    ''' the simulation of gas'''
    def __init__(self, r_ball=0.1,s_ball=20,r_container=10, num_b=50,mass_b=1e-23,
                 fps=200,diatomic=False,num_b2=10,r_ball2=0.5,mass_b2=2e-23,
                 Brownian=False):
        '''initialise the a single ball in container'''
        self.__rc=r_container #radius of container
        self.__c=Ball(1e15,self.__rc,[0,0],[0,0], container=True) #container
        self.__rb1=r_ball #the radius of small balls inside container
        self.__mb=mass_b # the mass of balls
        self.__q1=num_b #quantity of balls
        self.__ti=1./fps  #time interval between two frames/ 100fps
        self.__time=0 #time elapsed
        self.__spd=s_ball#the mean speed of balls in one dimension
        self.__i=0 #the cumulative inpulse counter used to compute pressure
        self.__pt=0 #the time counter use to calculate pressure
        self.__P='TBD' #the pressure, tbd at the begining
        self.NC=0 #the number of collision used in method simulation
        self.__brw_r=1 #the radius of brownian ball
        
        pos1=[] # the list of positions of ball
        
        if Brownian ==True:
            #if want to simulate brownian motion
            brw_r=1 #radius of big particle
            brw=Ball(1e-22,self.__brw_r,[0,0],[0,0],clr='brown')#bigger and heavier
            
        while len(pos1) != self.__q1:
            #to add balls until reach the required quantity
            p1=[random.uniform(-(self.__rc-self.__rb1),self.__rc-self.__rb1),
                random.uniform(-(self.__rc-self.__rb1),self.__rc-self.__rb1)]
            
            overlap=False
            # the state if the new ball overlap with original balls
            if Brownian==True:
                #check if we want to investigate Brawnian motion
                if  ((p1[0])**2+(p1[1])**2)**0.5 < brw_r+self.__rb1+0.1:
                    #check if small balloverlap with the big ball
                    overlap=True
            
            for i in range(len(pos1)):                
                if ((p1[0]-pos1[i][0])**2+(p1[1]-pos1[i][1])**2)**0.5 < 2*self.__rb1+0.1:
                    overlap=True
                    #check if small balls overlap with each other
                    
            if (p1[0]**2+p1[1]**2)**0.5 >10-self.__rb1:
                #finally, check if the ball overlap with container
                pass
                #if yes, pass the ball without appending
                
            elif overlap :
                #if balls overlap,pass without appending
                pass
            else:
                pos1.append(p1)
                #if the ball dont overlap with container and other balls,append
                
        pos2=[]#the list of positions of second kind of atoms
        
        if diatomic ==True:
            #only add second kind of balls if we want to check diatomic gas
            self.__q2=num_b2 #the number of the second kind of balls
            self.__rb2=r_ball2 #the radius of the second kind of balls
            self.__mb2=mass_b2#the mass of the second kind of balls
            
            while len(pos2) != self.__q2:
                p2=[random.uniform(-(self.__rc-self.__rb2),(self.__rc-self.__rb2)),
                    random.uniform(-(self.__rc-self.__rb2),(self.__rc-self.__rb2))]
                
                overlap=False
                # the state if the new ball overlap
                
                for i in range(len(pos2)):
                    if ((p2[0]-pos2[i][0])**2+(p2[1]-pos2[i][1])**2)**0.5< 2*self.__rb2+0.1:
                        #check if the new kind of balls overlap
                        overlap=True
                        
                if Brownian ==True:
                    if (p2[0]**2+p2[1]**2)**0.5 <self.__brw_r+self.__rb2+0.1:
                        #check if new kind of ball overlap with brownian ball
                        overlap=True
                    
                if (p2[0]**2+p2[1]**2)**0.5 >10-self.__rb2:
                    #check if the new kind of balls overlap with container
                    overlap=True
                    
                for i in range(len(pos1)):
                    if ((p2[0]-pos1[i][0])**2+(p2[1]-pos1[i][1])**2)**0.5<self.__rb1+self.__rb2+0.1:
                        #check if the new ball overlap with first kind of balls
                        overlap =True
                
                if overlap ==False:
                    pos2.append(p2)
                    
        pos=pos1+pos2#the totoal position list
        self.__q=len(pos) #the number of balls
        
        self.__pos=np.array(pos)
        
        vel=np.array([random.uniform(-2*self.__spd,
            2*self.__spd) for i in range(2*self.__q)]).reshape(self.__q,2)
            #make the velocity array, the mean speed is self.__spd roughly
                    
        vel[:,0] -= vel[:,0].mean() #make sure mean x velocity is 0
        vel[:,1] -= vel[:,1].mean() #make sure mean y velocity is 0
        self.__vel=vel #velocity of balls

        self.__bs=[self.__c] # the balls
        for i in range(self.__q1):
            self.__bs.append(Ball(self.__mb,self.__rb1,self.__pos[i],self.__vel[i]))
            #append first kind of balls
        
        if Brownian==True:
            self.__q +=1
            self.__bs.append(brw)
            #append the big particle
            
        if diatomic==True:
            for i in range(self.__q2):
                self.__bs.append(Ball(self.__mb2,self.__rb2,self.__pos[self.__q1+i],
                                      self.__vel[self.__q1+i],clr='green')) 
                #append the second kind of balls
            
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
        if p < 1e-18:
            #avoid rounding error of 0 in pythhon flash 
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
        temp=mean_ke/k
        return temp
    
    def time(self):
        '''display the time elapsed'''
        return self.__time
    
    def simulation(self):
        '''
        a very simple simulation of the motion of balls without animation
        used to compute numerous balls velocities distribution
        '''
        p_tc=[] #all calculated time to collide between two single balls
        bs_c=self.balls()[:] #a copy of the balls, the ball to collide
               
        for i in range(self.__q):
            b1=bs_c.pop(0)
            
            for j in range(len(bs_c)):
                dt=b1.time_to_collision(bs_c[j])
                p_tc.append((i,j+i+1,dt))

        c_set=min([i for i in p_tc] ,key=lambda x: x[2])
        #the tuple of collide set,[0] is self, [1] is other, [2]is next time to collision
        
        tc=c_set[2]
        
        for b in self.__bs:
            b.move(tc)
            
        self.balls()[c_set[0]].collide(self.balls()[c_set[1]])   
        self.NC +=1
        
    def init_figure(self):
        '''
        Initialise the container diagram and add it to the plot
        '''
        
        ax=ax1
        BigCirc = plt.Circle((0,0), self.__rc, ec = 'b', fill = False, ls = 'solid')
        ax.add_artist(BigCirc)
        
        self.__p_text=ax.text(-self.r_container()-2.9,
                               self.r_container()-1,
                               'Pressure=TBD')   
        #pressure indicator
        
        self.__energy_text=ax.text(-self.r_container()-2.9,
                                   self.r_container()-2,
                                     f"KE.={self.energy():.3g}J")
        #energy indicator
        
        self.__temp_text=ax.text(-self.r_container()-2.9,
                                 self.r_container()-3,
                                 f'Temp={self.temp(): .3g}K')
        #temperature indicator
        
        self.__momentum_text = ax.text(-self.r_container()-2.9,
                                       self.r_container()-4,
                                       f"p={self.momentum() :.1f}"+r'$kgms^{-1}$')
        #momentum indicator
        
        self.__time_text=ax.text(-self.r_container()-2.9,
                                       self.r_container()-5,
                                       f't={self.__time: .1f}s')
        #time indicator
        
        self.__text0=ax.text(-self.r_container()-2.9,
                              self.r_container()-6,
                              f"f={0}")
        #framenumber indicator
   
        self.__patches=[self.__text0,
                        self.__energy_text,
                        self.__momentum_text,
                        self.__time_text,
                        self.__temp_text,
                        self.__p_text,
                        ]
        #append all indicators to patches
        
        #add the patches of the balls to the plot
        for b in self.__bs:
            pch = b.get_patch()
            ax.add_patch(pch)
            self.__patches.append(pch)
            #append balls to patches
        
        
        return self.__patches
    
    def next_frame(self,framenumber):
        '''
        Do the next frame of the animation.
        This method is called by FuncAnimation with a single argument
        representing the frame number.
        Return a list or tuple of the 'patches' being animated.
        '''
        #here to find the next time to collide and update N
        p_tc=[] #all calculated time to collide between two single balls
        bs_c=self.__bs[:] #a copy of the balls, the ball to collide
               

        for i in range(self.__q):
            b1=bs_c.pop(0)
            
            for j in range(len(bs_c)):
                dt=b1.time_to_collision(bs_c[j])
                p_tc.append((i,j+i+1,dt))

        c_set=min([i for i in p_tc] ,key=lambda x: x[2])
        #the tuple of collide set,[0] is self, [1] is other, [2]is tc
        
        tc=c_set[2]
        
        if self.__pt >= 0.5:
            #when time counter pt beyond the time interval
            self.__P=self.__i/self.__pt/(2*np.pi*self.r_container())
            #compute the pressure over the period of t
            self.__pt=0 #reset time counter pt
            self.__i=0   #reset inpulse counter i 

        self.__text0.set_text(f'f={framenumber}')
        self.__energy_text.set_text(f"K.E={self.energy():.3g} J")
        self.__momentum_text.set_text(f"p={self.momentum()}"+r'$kgms^{-1}$')
        self.__time_text.set_text(f't={self.__time : .1f}s')
        self.__temp_text.set_text(f'Temp={self.temp() :.3g}K')
        
        if self.__P=='TBD':
            self.__p_text.set_text(f'Pressure={self.__P}')
        else:
            self.__p_text.set_text(f'Pressure={self.__P :.3g}Pa')   
                                     
        if tc >= self.__ti:

            self.__N=int(tc /self.__ti)
            # counter, also is the step need to take to collide
            self.__dt = tc/self.__N  
            
            self.__patches=[
                            self.__text0,
                            self.__energy_text,
                            self.__momentum_text,
                            self.__time_text,
                            self.__temp_text,
                            self.__p_text,
                            ]    
            
            for b in self.__bs:                
                self.__patches.append(b.get_patch())

        else:
            self.__N = 1
            self.__dt = tc   
            
        for b in self.__bs:
            b.move(self.__dt)
                
        self.__time += self.__dt #increase time
        self.__pt += self.__dt #increase pressure time interval counter

        self.__N -= 1
        if self.__N == 0:            
            if self.__bs[c_set[0]].container():
                #when ball collide with counter
                I=self.__bs[c_set[1]].cmmt()
                #computhe the change in momentum in ball
                self.__i += I
                #add the inpulse to the the cumulative inpulse counter
                
            elif self.__bs[c_set[1]].container():
                I=self.__bs[c_set[0]].cmmt()
                self.__i += I
                
            else:
                pass
            
            self.__bs[c_set[0]].collide(self.__bs[c_set[1]])
            
        return self.__patches
    
    def init_hist(self):
        '''initialise the histgram'''

        v=list(b.speed() for b in self.__bs)
        n, _ = np.histogram(v, bins, normed=False)
        for rect, h in zip(patches, n):
            rect.set_height(h)
        
        self.hist_ix=ax2.get_xlim()
        #the initial xlim of hist is determined by python at the begining
        self.hist_iy=ax2.get_ylim()
        #the initial ylim of hist is determined by python at the begining
        self.__var_text=ax2.text(self.hist_ix[0]+0.1,
                                 self.hist_iy[1]-2,
                                 f'variance={self.var()}')
        
        self.__temp_text=ax2.text(self.hist_ix[0]+0.1,
                                 self.hist_iy[1]-3,
                                 f'T={self.temp()}K')
        
        self.__text0=ax2.text(self.hist_ix[1]*0.02,
                              self.hist_iy[1]*0.7,
                              f'framenumber')
       
        patches.append(self.__var_text)
        patches.append(self.__temp_text)
        patches.append(self.__text0)
        
        i_patches=patches[:]
        #need to delete original indicators after return pathces, so use a copy
        
        patches.pop()
        patches.pop()
        patches.pop()
        #delate indicators
        return i_patches
    
    def update_hist(self,framenumber,update_time=5):
        '''
        Do the next frame of the animation of histogram, update patches every
        updata time.
        '''
        #here to find the next time to collide and update N
        p_tc=[] #all calculated time to collide between two single balls
        bs_c=self.__bs[:] #a copy of the balls, the ball to collide

        for i in range(self.__q):
            b1=bs_c.pop(0)
            
            for j in range(len(bs_c)):
                dt=b1.time_to_collision(bs_c[j])
                p_tc.append((i,j+i+1,dt))

        c_set=min([i for i in p_tc] ,key=lambda x: x[2])
        #the tuple of collide set,[0] is self, [1] is other, [2]is time to collide
        
        tc=c_set[2]            
                                
        if tc >= self.__ti:

            self.__N=int(tc /self.__ti)
            # counter, also is the step need to take to collide
            self.__dt = tc/self.__N 

        else:
            self.__N = 1
            self.__dt = tc

        for b in self.__bs:
            b.move(self.__dt)
            
        self.__N -= 1

        if self.__N == 0:                        
            self.__bs[c_set[0]].collide(self.__bs[c_set[1]])             
        
        if framenumber%update_time != 0:
            #if framenumber is not multiple of required updata time,
            #no change in frames
            pass
            
        else:
            patches.pop() 
            patches.pop() 
            patches.pop()
            #remove the old indicators
            
            v=list(b.speed() for b in self.__bs)
            n, _ = np.histogram(v, bins, normed=False)
            for rect, h in zip(patches, n):
                rect.set_height(h)
            #add histogram into patches
            
            print(f'variance={self.var()}')
            
            ax2.set_xlim(0,self.hist_ix[1]*1.05)
            ax2.set_ylim(0,self.hist_iy[1]*1.2)
            
            self.__var_text=ax2.text(self.hist_ix[1]*0.02,
                                     self.hist_iy[1]*1.2-2,
                                     f'var={self.var()}')
            #variance text
            
            self.__temp_text=ax2.text(self.hist_ix[1]*0.02,
                                     self.hist_iy[1]*0.9,
                                     f'T={self.temp()}K')
            #temperature text
            
            self.__text0=ax2.text(self.hist_ix[1]*0.02,
                                     self.hist_iy[1]*0.7,
                                     f'f={framenumber}')
            #framenumber text
           
            patches.append(self.__var_text)
            patches.append(self.__temp_text)
            patches.append(self.__text0)
            #add indicators
        
        return patches

if __name__ == "__main__":
        
      #animation of motion of snooker
    if animate_2d == True:
        
        movie = Gas(Brownian=ifbrow,diatomic=ifdiatm)
        fig1 = plt.figure(1)
        ax1 = plt.axes(xlim=(-movie.r_container()-3, movie.r_container()),
                      ylim=(-movie.r_container(), movie.r_container()))
        ax1.axes.set_aspect('equal')  
            
        anim = animation.FuncAnimation( fig1, 
                                        movie.next_frame, 
                                        init_func = movie.init_figure, 
                                        interval = 10,
                                        blit = True)
        plt.show()
        
    #an animation histogram, shows vel distribution, which updata every 5 frames
    if histogram_2d == True:
        live_hist = Gas(0.01,20,10,num_b=500)
        
        fig2 = plt.figure(2)
        
        ax2 = plt.axes()
    
        ax2.axes.set_aspect('equal')  
        v=list(b.speed() for b in live_hist.balls())
        n, bins, patches = plt.hist(v,100, facecolor='green',alpha=0.75)
    
        ani = animation.FuncAnimation(fig2,
                                      live_hist.update_hist,
                                      init_func=live_hist.init_hist,
                                      blit=True,
                                      interval=1)
        plt.show()

    #Maxwell Boltzmann distribution investigation
    if M_B_distribution_test ==True:
        mb_test=Gas(r_ball=0.01,num_b=3000,s_ball=20)
        
        while mb_test.NC != 50000:
            mb_test.simulation()
            print(mb_test.NC)
            
        fig3=plt.figure(3)
        ax3=plt.axes()
        ax3.axes.set_aspect('equal')
        v=list(b.speed() for b in mb_test.balls())

        with open('data.txt','w') as f:
            for i in v:
                f.write(str(i)+'\n')
            
        plt.hist(v,100, facecolor='green',alpha=0.75)        
        plt.show()
        print(f'variance={mb_test.var()}')
        print(f'T={mb_test.temp()}K')



    