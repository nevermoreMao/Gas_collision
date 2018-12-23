In twodthermo simulation, there are five switches at the beginning which allow you to use different code.

M_B_distribution_test is used to collect the data velocities distribution of 3000 balls, but 50000 collisions simulated to approach the Boltzmann distribution.
It take more than one days to simulate collisions. What's more, !!!it is extremely likely to make your PC crash. So please dont open it unless you really want to!!!!!!!

animation_2d is switch of animation of particles motion.
There are several indicators in the animation, they are updated every frame apart from pressire.
Pressure is updated every 0.5s, it is 'TBD' at the first 0.5 second, so please be patient to wait pressure figure.

histogram_2d is animation of velocities distribution of 500 balls, it is updated every five frames. There is a big lag but won't make your PC crash I promise.
In real life, it updates per 3-5 seconds. Be patient...

ifbrow and ifdiatm are switches for brownian and diatomic gas simulation. Make sure animate_2d is opened before you want to both these two. You can open both of them at same time

In threedthermo, M_B_distribution_test is as unfriendly as that in twodthermo.
pressure_test will compute five pressures and find the mean value. std and temperature are printed as well.

Finally, dont open M_B_distribution_test.