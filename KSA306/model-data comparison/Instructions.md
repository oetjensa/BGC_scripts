**Comparison of Modelled and Observed Marine NPP**

**Introduction:** Primary Production is the photosynthetic transformation of inorganic carbon (e.g. CO2) into organic matter that can be used to build biomass or be broken down to create energy. Net Primary Production is the net production of biomass by autotrophic organisms after subtracting out the amount of biomass they use to for their own energy needs (i.e. autotrophic respiration).

Marine algae (i.e. phytoplankton) are responsible for about half the NPP on Earth despite on comprising only about 1% of the total biomass at any given time! That’s because -- unlike trees on land-- they live in a highly dynamic fluid environment with a very high turnover time. They grow fast but die fast too. Ultimately, phytoplankton NPP constitutes the base of the marine food-web and is tightly related to strength of the biological pump, which can create a DIC gradient in the ocean and lead to order 100 ppm changes in the atmospheric CO2 concentrations.

Thus, understanding how NPP will change in future climate states is critical to understanding Earth’s capacity to feed a growing population and buffer a changing climate. Earth System Models (ESMs) used by the IPCC to anticipate different climate scenarios account for this by including a marine biogeochemical component model that simulates marine NPP. Unfortunately, there is a huge amount of uncertainty in future projections of NPP (Tagliabue et al., 2021). Even under identical emissions forcing scenarios, state-of-the-art Earth system models can’t even agree on the direction(!), no less the magnitude, of changes to NPP in a warmer ocean (Figure 1). While it is impossible to know which model prediction is ‘right’, the first step is to evaluate how well they can create present day conditions. 

In this lab you will pick two models and one remote sensing product to compare the seasonal evolution and spatial distribution between the models and the observations.

![image](https://github.com/user-attachments/assets/c5d4016f-0e4e-406f-957b-af8bb15f34b3)
Figure 1. Projections of physical (MLD, SST) and biological (Biomass, NPP) variables for all 11 models considered in this lab under SSP585 forcing.


**Section 1:** Pick and Download Your Data 

Pick your Models. What models do you want to look at? I processed the NPP climatologies from 11 prominent ESMs. Each one is averaged from about 1750 to 1900, long enough average out ENSO variability, but stopping before the anthropogenic signal gets too big. 
These are all available to download on the Mylo webpage. If you want to download data from different models or variables for your own research, check out the information in Appendix 1 on MyLO.

![image](https://github.com/user-attachments/assets/08e41738-d201-44e4-b850-18001e5a15b2)
Table 1. List of BGC models and their associated Earth System Model. 

The table above provides a list of all the available models. Each ESM is built out of several different model components: one for each part of the earth system: the marine BGC, the ocean physics, the atmosphere, etc. The first column in Table 1 lists the name of the BGC model component, the second lists the broader Earth system model, and the fifth lists the organization that runs it. The third and fourth give details on the specific simulation and ensemble member. In this case all simulations were forced with historical atmospheric CO2 concentrations. The ensemble member relates to small variations in the initial conditions the run was initialized with. Often models will be run from many different sets of initial conditions to understand ‘natural’ variability. If you are interested in any more details on the models or experiments the citations included refer to the reference list found in Rohr et al (2023).

________________________________________________
Question 1: Why do you think I have provided two outputs for the BGC model called PISCESv2? How might do you think they may or may not be different?
________________________________________________
You might wonder how all these BGC models differ. There are quite a lot of ways, ranging from nutrients included to the way particles sink (Seferian et al., 2020). But the way that might be the most important for NPP is the composition of the food web. That is, how many types of phytoplankton and zooplankton they include, and how they interact. Check out the schematic below in Figure 2 to get an idea of the different food webs in the different BGC models.

Models with multiple types of phytoplankton typically differentiate them based on their growth rates and light and nutrient requirements. Small phytoplankton typically grow really fast, need lots of light and not so many nutrients. Big phytoplankton (like diatoms) grow slower, need more nutrients, but can survive at lower light levels. 

Models with multiple types of zooplankton typically differentiate them based on what they like to eat and how fast they can eat it. Smaller zooplankton typically prefer small phytoplankton and graze on them really fast. Larger zooplankton typically can consume the bigger phytoplankton but must eat them at a much slower rate.

________________________________________________
Question 2: Based on the descriptions above, in a model like CanOE with 2 types phytoplankton and 2 types zooplankton, a small and large of each, how do you think the populations of each might change in a changing climate.


Question 3: Have a look at the schematic below in Figure 2. Based on the described food webs which models do you think simulate food-webs that respond most accurately to climate change and why?
________________________________________________


![image](https://github.com/user-attachments/assets/431a4223-ff86-4787-95ef-309cc2ffb0cc)
Figure 2. A) The marine food webs represented in 10 CMIP6-class BGC models are presented in clockwise order of increasing complexity. Next to the name of each BGC model is the number of parameters required to describe grazing. Grazing relationships (arrows) are solid for single-prey responses, dashed for multi-prey responses with fixed preferences, and dotted for multi-prey responses with active switching. Red arrows indicate temperature sensitivity. Green, red/orange, and purple color schemes refer to models with 1, 2, or actively switching zooplankton, respectively. Hue scales qualitatively with complexity. PFTs have been generalized into small (nano-, small, non-diatom, or nanoflagelate), large (large, diatoms), or diazotrophs for phytoplankton (P) and small (micro-, small), medium (meso-, medium), or large (macro-, large) for zooplankton (Z). Bacteria (B) and Detritus (D) are included when available as prey. B) In the inner panel the Prescribed Grazing Index is plotted for different models and observations. This metric reflects how fast the average zooplankton in that model or observed functional group grazes on the global median plankton population. Figure from Rohr et al (2023).

**Pick your Remote Sensing Data.** 
What Remote Sensing Product do you want to look at? VGPM and CbPM are two different algorithms used to compute NPP from remote sensing data and various empirical relationships. 
I have created ~10 year climatologies for each, but check out Appendix 2 to learn more how to download them yourself. As for the difference between the models?
To estimate NPP, the depth-integrated rate of productivity, you need three things: a rate term (i.e. how fast phytoplankton are dividing), a concentration (how many are doing the dividing), and a volume function that tells you how the surface concentration changes with depth (as the satellite can only see the surface concentration). See Table 2 (also on Mylo) for a brief description of how VGPM and CbPM differ in these regards.


![image](https://github.com/user-attachments/assets/0ecd4304-997c-4ac4-977b-8740045e640c)
Table 2. Descriptions of two prominent Remote Sensing NPP 'models' 


________________________________________________
Question 4: What is the key satellite measurement utilized in CbPM that differentiates it from VGPM? 
________________________________________________

Setup and Download. Once you have picked your data sources, download your:   

2 selected BGC model outputs, 1 selected remote sensing dataset and with ALL matlab scripts provided in the ‘Data and Files’ section of the MyLo Webpage. 

Make sure to put all these files in the same folder. Also make sure that the matlab script you download didn’t end up with a funny name in your download folder (that is no spaces or paraenesis; delete them if needed). 

Then you can open the main matlab script for this lab: QMS_Model_Eval.m and modify and execute ‘Section 1’. Make sure to read all the comments in the script as you move along, as some of the line of code need to be changed! 

Lines of code that need to be changed are bracketed as:

% !!!!!!!! MODIFY This CODE !!!!!!!
  CODE to Modify
% !!!!!!!! MODIFY This CODE !!!!!!!

Replace ‘XXX’s text with your modified code. 

Finally, to execute a section of script either click ‘Run Section’ in the ‘Editor’ tab up top or highlight the code, right click, and select ‘Execute Section’.

Plot your data from one model and one month. Each model output is saved as a spatially resolved, depth integrated, monthly climatology in a 3D matrix.


Note: You will need the ‘Mapping Toolbox’ to make these plots. If you get error regarding the ‘axesm’ command you may need to download it. From the top menu go to: Home -> Add-ons -> Get add-ons -> search ‘mapping toolbox’ -> Install -> log in with your math works credentials you made when you downloaded matlab with the university license. If this doesn’t work, there are some work arounds commented out in the script. 


________________________________________________
Question 5: What are the dimensions of the model out matrixes.
________________________________________________



________________________________________________
Question 6: Modify Section 1.1 of the script to specify and plot a model and month to plot. 
________________________________________________





Section 2: Plot the global, annually integrated NPP distributions

Run Section 2.1 of the script to plot the global distribution of mean annual NPP. 
________________________________________________
Question 7: How do I compute the mean annual NPP in the code? What is a weakness with the approach? What would happen is you averaged the NPP_model_1 file across a different dimension? 

Question 8: How do I compute the globally integrated annual NPP in the code?

Question 9: Which of your models performs better at producing the mean annual NPP distribution? Can you make a plot to quantify this? If not, why not?
________________________________________________

Run ‘Section 2.2: Regrid’ of the code to interpolate the coarser model grid onto the finer resolution remote sensing grid in both space and time. This might take a little while (~10-15 min). While you wait start to have a think about Question 6. 


Question 10: Once your model data is regridded, plot the distribution of the normalized model bias for both of your models relative to the remote sensing data. Section 2.3 of the script will walk you through this, but you will have to compute the bias variables yourself and then make sensible decisions about what colorbounds and colormap to use to plot your data. 

Hint. Have a look at the colormaps available from cmocean online to help you pick one.

Hint. If you are unfamiliar with MATLAB, have a look at the ‘Mathematical Operations in Matlab’ Primer on the Mylo page. This outlines some basic mathematical commands in the MATALB that you can use in your calculations. 

Hint. Anytime you are making a plot, you can change the size of the font on the labels by increasing/decreasing the number the follow ‘fontsize’ in the script. Make sure your fontsize is always sensible. 



Section 3: Pick a region and look at the seasonal cycle
Instructions: Validating an entire model might be biting off a bit more than we can chew. Let’s focus on a smaller region of interest. Something big enough to average out the noise, but small enough to not average out the signal of interest. 

Here we will break things down by their seasonal average: December, January, February (DJF); March, April, May (MAM), and so on.  This lets us understand if the model is performing better or worse during any given part of the year.

________________________________________________
Question 11: Pick a region. What region did you pick? Why does it interest you? You can define the bounds of your region in Section 3.1. 


Question 12: Compute the seasonally, and regionally integrated NPP and add it to each subpanel by modifying the code in Section 3.2. Then plot it by running Section 3.2. Include your Figure.
________________________________________________

Still, there is a lot going on. Run Section 3.3 of the code to plot the spatially averaged seasonal cycle for each data set. Think about how they compare.  

On the top right I have included a set of summary statistics to help quantify your intuition. Have a look at the Description of Statistical Metrics Table online on the Mylo webpage for a definition of each statistic and the underlying math and interpretation

________________________________________________
Question 13: Which model recreates the regionally averaged seasonal cycle in the remote sensing record best? Support your conclusion statistically. Note, there may be trade-offs worth noting between various statistics. 
________________________________________________

It also important to think about the ecological significance of these time series. Phenology refers to the timing of biological cycles. With respect to phytoplankton productivity in the ocean, it typically refers to the seasonal timing of things like the initiation of the blooming period (i.e. when biomass begins to accumulate after winter) and the peak NPP.

________________________________________________
Question 14: Describe quantitatively how well the model phenology agrees with the observed phenology. A) Specifically, how well aligned in the timing of peak NPP? B) Can you say precisely when bloom initiation begins from these data sets alone? If not, how might you approximate it.
________________________________________________

Now, notice in the bottom right corner of the figure, below the summary statistics is a funny looking diagram. This is called a Taylor diagram. It is a handy way of comparing three statistics at once: the normalized, centred root mean square error (ncRMSE), the normalized standard deviation (nSTD), and the correlation coefficient (r). Note, I have highlighted all three in the summary table in the figure. 

•	The normalized standard deviation (nSTD) is plotted as the distance between the model data point and the origin of the graph. A value >1 (or <1) means the standard deviation of the model is greater (or smaller) than the standard deviation of the observations. That is, there is more or less variance. When normalized, a value of 1 means the model and obs have the same STD. Ecologically, this could imply the amplitude of the seasonal cycle is greater or smaller than that of the observations, respectively. 

•	The correlation coefficient (r) is plotted as the angle from the x-axis. A correlation coefficient closer to 1 means the model covaries more with the observations. In other words, the time series move up and down at the same time and by the same amount relative to their respective standard deviations. Ecologically, a strong correlation could imply they have similar phenologies. That is, NPP begins to increase and decrease at the same time of the year.

•	The normalized, centred root mean square error (ncRMSE), is plotted as the absolute distance between the model data point and the observed data point. Further away means more error. Importantly though, the cRMSE is centered, meaning the mean value of each seasonal cycle has remove. My definition, this means it does not reflect the model bias (i.e. the difference in the means). Thus, the ncRMSE reflects the combination of discrepancies in the size of the variance (i.e. nSTD) and alignment of variance (i.e. r), but not any differences in the mean values. When normalized, the size of the error is expressed relative to the size of the observed STD. Ecologically, this is a useful summary statistic of how well the patterns of variance match between model and observations. Importantly though, it says nothing about bias or differences in the mean state. 

Finally, note the location of the point describing the observed timeseries. It always falls on the x-axis because its correlation with itself is always 1. Here, because we’ve normalized the data, it also has a nSTD of 1. This is because its STD is divided (i.e. normalized) by itself. 

To learn more about Taylor Diagrams check out this helpful primer. Note, in their example, they don’t normalize the cRMSE or STDs, meaning the observed data point doesn’t fall on a value of 1. The benefit of normalize the data is that you can plot different metrics with different units on the same plot.

________________________________________________
Question 15: Despite having two dimensions, three statistics are plotted in the Taylor Diagram: the normalized standard deviation, the correlation coefficient and the ncRMSE. How is this possible? Hint. Check out the equations on the second page of the primer provided above. 
________________________________________________


Finally, you may be interested in spatial variability in model performance. For example, if your region is large and includes a highly dynamic boundary current you may notice substantial differences in model performance depending where you look. 

Because we have regridded the model and observed data sets it is possible to compute identical statistics to the ones described above, but instead of comparing the spatially averaged timeseries, we can compare the timeseries individually at each grid cell and plot their performance.

Run Section 3.4 and 3.5 to compute and plot the spatial distribution of the three ‘Taylor Statistics’ described above. 

________________________________________________
Question 16: Where does your model perform best and why? In the regions where it performs worst does the model appear to mostly be misrepresenting the phenology (i.e. the timing or alignment) of the seasonal cycle or the amplitude (i.e. the amount of variance) of the seasonal cycle?
________________________________________________


________________________________________________
Bonus Question: Can you make a new plot showing the bias in the magnitude of the peak annual NPP? 




