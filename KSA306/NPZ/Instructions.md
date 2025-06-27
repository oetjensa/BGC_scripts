**Examples of nutrient-phytoplankton-zooplankton (NPZ) modelling**

**Goal:** To provide some experience in running, modifying and interpreting a simple Nutrient-Phytoplankton-Zooplankton model.

**Important:** Saving files on the lab computers may be problematic. This lab might work best if you work entirely from a USB stick, maybe with a dedicated folder. Or if you are comfortable working in a networked folder, do that.

**Introduction:** One of the goals of biological oceanography is to understand the interactions between nutrient availability, phytoplankton biomass (which depends on the phytoplankton growth rate, mortality, grazing of phytoplankton by zooplankton, physical mixing and sinking) and zooplankton biomass (which depends on their grazing of phytoplankton and mortality). Here we use a basic nutrient-phytoplankton-zooplankton (NPZ) model of the upper water column (a temperate ocean mixed layer) based on a 1986 paper by Peter Franks – see franks_npz_model.pdf.

In this model, nutrients (N, probably mostly as NO3) are taken up by phytoplankton and converted into phytoplankton biomass (or ‘stock’). Phytoplankton are grazed (eaten) by zooplankton and converted into zooplankton biomass but the process is not 100% efficient, there are losses due to metabolism. These losses are excreted back into the dissolved nutrient pool (as NH4) and are available to phytoplankton. Both phytoplankton and zooplankton die and decay and are remineralised back into the dissolved nutrient pool which is again available to phytoplankton. Note that the system is closed – there is no net flux of anything in or out (until we implement it later in the exercise). All three ‘pools’ (N, P and Z) are expressed in units of the concentration of N [M or its equivalent unit mmol m-3]. This is known as the ‘currency’ of the model.

Note: For validating model output, N is an inconvenient term because there are few observations of particulate organic N (PON). Chlorophyll would be better because there are abundant chlorophyll observations. So some modellers use a N:chl conversion factor and compare phytoplankton stocks with satellite or in situ chlorophyll observations.

The model is described by a series of equations that describe how much is added or subtracted from each pool (P, Z and N) at each timestep. Each addition or subtraction corresponds to a physical process (i.e. photosynthesis or grazing). Note, because mass is conserved any addition to one pool must correspond to a subtraction from another. 

For this simple model, we can write the equations out in words like this:
Change in P = + nutrient uptake – P mortality (i.e. non-grazing death) – grazing

Change in Z = + grazing efficiency *Grazing – Z mortality (i.e. all death)

Change in N = – nutrient uptake + (1-grazing efficiency) * Grazing + P mortality + Z mortality 

Where,

Nutrient uptake is equal to the rate of photosynthesis (u) multiplied by the amount of phytoplankton (P). 

The rate of photosynthesis (u) is equal to a maximum phytoplankton growth rate (umax) scaled by a nutrient limitation term (LimN). In more complex models rate of photosynthesis is often also scaled by a temperature and light limitation term. 

The nutrient limitation term (LimN) is equal to a Michelis-Menton style function of the nutrient concentration (N). It starts at 0 when there are no nutrinent, increases rapidly as nutrients increase, then saturates toward 1 once nutrients become no longer limiting. The half saturation concentration for nutrient uptake (KN) determines at what concentration nutrients become limiting. Higher values means nutrients are more limiting. 

The grazing efficiency (γ) is equal to a single parameter value that expresses what fraction of phytoplankton grazed actually get used to ‘build’ zooplankton biomass. The rest is either not digested or metalized for energy. This portion is implicitly remineralized and put back into the inorganic nutrient pool. 

Grazing (G) is equal to the rate of grazing (g) multiplied by the amount of zooplankton (Z). 

The rate of grazing (g) is equal to the maximum grazing rate (gmax) scaled by a function of the phytoplankton population. Although this function looks mathematically different than the nutrient limitation term, it is qualitatively similar. Each individual zooplankton can graze at faster rate when there are more phytoplankton and it is thus easier to find and capture them. 

The P and Z mortality terms are equal to a mortality rate (mP, mZ )  multiplied by their respective populations (P, Z). 

All together, these relationships can be written mathematically as: 
![image](https://github.com/user-attachments/assets/ce91180c-944d-4770-a2f7-531d0ced016e)



The variables (i.e. values that measure a physical quantity and change in time, aka tracers) and parameters (i.e. values the described the assumed rate of different processes) are summarized in the tables below. Note model variables (or tracers) are then physical quantities being modeled while parameters describe the assumptions the modeler has made about the processes that relate them.

<img width="659" alt="image" src="https://github.com/user-attachments/assets/d0c3ec79-908d-435c-838a-6aca7661b83e" />


<img width="554" alt="image" src="https://github.com/user-attachments/assets/26afaa6b-a49b-423b-9bd1-17408eb9acd6" />



The scenario we are simulating here is much like an idealized spring bloom in, say, the North Atlantic or Tasman Sea. We are in a mid-to-high latitude ocean around the beginning of spring when the water column has just stratified, nutrients are at moderate to high levels from winter mixing, and both phytoplankton and zooplankton biomass are low to begin.

The exercise consists of several parts. First, we’ll run the basic model, then we’ll tweak the parameters in different ways, look at the output and answer some questions.

1.	Get the notebook and save it somewhere sensible (see the note above about being organized). Confirm all initial conditions and parameter values are consistent with Tables 1 and 2. Run the script and look at your figure1.

2.	A phytoplankton growth rate (umax) of 2 day-1 is not very realistic. It’s too high. A more realistic value would be 0.69 day-1 which corresponds to one doubling per day (0.69 = ln(2)). Also, an initial nutrient (nitrate, NO¬3) concentration of 1.6 M is too low for our theoretical north Atlantic, let’s try 10.6 M. Change these values in your script. Change the figure name in the last line to figure2.jpg and run the script. 
______________________________________________________
Question 1. What are the main differences, if any, between figure1 and figure2 for all three pools of nitrogen (N, P, Z)? Explain how those differences came about, or not.
_____________________________________________________

3.	The Matlab script includes the option to set a minimum amount phytoplankton concentration (P0, or ‘P-zero’) that must be present before they start to be grazed. This can be invoked by setting P0 to a non-zero value. Set it to 1.0, change the figure name to figure3.jpg and run the script. 

__________________________________________________
Question 2. What is the main difference between figure2 and figure3 and why? What ecological phenomenon is P0 trying to simulate?
__________________________________________________

4.	The large amount of N that is regenerated as grazing occurs is not realistic. It happens because of the closed system. So, let’s move away from a model that operates in a closed upper mixed layer and allow some interaction with the ocean below. We will let some of the dead P and Z (mPP and mZZ) sink. We’ll also let some of the deep water mix into the surface waters, and we’ll lengthen the duration of the simulation. So, do this: (1) change ndays to 200, (2) turn mixing on by setting  mixingOn to ‘true’ in line 25 and (3) change the figure name to figure4 and run the script.

__________________________________________________
Question 3. In answering these questions, pay careful attention to (1) the initial value of the parameter Mix (line 17), and (2) the way mixing is implemented in lines 35-37 and 52. In this new configuration, how much dead phytoplankton, wasted grazing and dead zooplankton is being remineralized? Hint. Look at the ‘Mixing Parameters’. What is happening to the rest? Why is it possible for N, P and Z to reach something of a steady state? What happens around day 120 and what do you think this is trying to simulate? Think about the annual cycle presented in class, and consider varying some parameters to understand the sensitivity of the model to their numerical values.
__________________________________________________


5.	The final part of the lab is a bit experimental. In the new code we have added a second type of phytoplankton, which is functionally different from the first. By functionally different we mean it has different physiological and ecological properties that are expressed via different parameters in the model. Notice the is now a separate umax, KN, and gmax for each type of phytoplankton. This means they have different nutrient requirements, growth rates and vulnerability to predation. 

These sorts of physiological difference are common throughout the ocean. For example, smaller phytoplankton with low KN can growth faster in low nutrient conditions and thus typically dominate I the gyres where nutrients concentrations are low (as there is little upwelling or vertical mixing to supply new nutrients). Larger phytoplankton, like Diatoms, often have larger KN and can only survive in nutrient rich environments like the Southern Ocean. These species are often adapted with other tradeoffs, such as shells that protect them against predation, allowing them to outcompete smaller phytoplankton in places with high nutrients. 

Note, if one phytoplankton is consistently better adapted (say lower nutrient requirements, faster growth rates, and more protections against grazing) then it will outcompete the other and drive it coextinction. However, if there are tradeoff if their functional properties it may be possible for them to coexist.

Take the Franks2spp.m script and try to create a scenario where both phytoplankton species coexist. That is, they’re not both zero or maxed out for the whole simulation. 
__________________________________________________
Question 4. What values of KN, umax, and gmax did you have to use for each species to achieve coexistence? 
__________________________________________________

6.	In lectures we discussed phytoplankton functional types on a very basic level. That is, we talked about:
a)	Diatoms need Si to make their glass shells in addition to the ‘regular’ macronutrients N and P.
b)	Diazotrophs like trichodesmium. They have specific enzymes which can break down dissolved gaseous N2 into a usable form of N.
c)	Coccolithophorids like Emiliania huxleyi. They make CaCO3 shells. The process of CaCO3 precipitation impacts ocean alkalinity and pCO2.

Realistic, successful models need to simulate these processes and you will learn more about this later in the week. The model we have used today is extremely simple.

__________________________________________________
Question 5. How would you take our Franks2spp.m script and make the 2nd phytoplankton species a diatom? Consider, for example:

•	What key variable (i.e. tracer) is missing from the current model to accurately simulate the constraints on Diatom growth?


•	In turn, what addition variables (i.e. tracers) and/or equations would you need to add? 


•	How would parameter values (e.g. KN, umax, and gmax) compare to those of small phytoplankton? Hint: See ‘The Representation of Phytoplankton slide in the Bonus ‘Introduction to BGC Modelling Lecture’ from Week 1.
_______________________________________________
