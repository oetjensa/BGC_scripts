'''
%  Originally Created by Guillaume Maze on 2008-10-28 in Matlab.
% Rev. by Guillaume Maze on 2010-02-10: Add NaN values handling, some checking
% in the inputs and a more complete help
% Copyright (c) 2008 Guillaume Maze. 
% http://codes.guillaumemaze.org
'''

import numpy as np

def allstats(Cr, Cf):
    """
    Compute statistics from 2 series considering Cr as the reference.
    
    Inputs:
        Cr and Cf: 1D numpy arrays of the same length, possibly containing NaNs.
    
    Outputs:
        STATM: A 4x2 array containing the statistics:
            1. Mean
            2. Standard Deviation
            3. Centered Root Mean Square Difference
            4. Correlation Coefficient
    """
    # Ensure inputs are numpy arrays and convert to column vectors (if not already)
    Cr = np.asarray(Cr).flatten()
    Cf = np.asarray(Cf).flatten()

    # Check that the arrays are the same length
    if len(Cr) != len(Cf):
        raise ValueError('Cr and Cf must be of the same length')
    
    # Remove NaN values
    valid_indices = ~np.isnan(Cr) & ~np.isnan(Cf)
    Cr = Cr[valid_indices]
    Cf = Cf[valid_indices]
    
    N = len(Cr)
    
    # Compute Mean
    mean_Cr = np.mean(Cr)
    mean_Cf = np.mean(Cf)
    
    # Compute Standard Deviation
    std_Cr = np.sqrt(np.sum((Cr - mean_Cr)**2) / N)
    std_Cf = np.sqrt(np.sum((Cf - mean_Cf)**2) / N)
    
    # Compute Centered Root Mean Square Difference (RMSD)
    rmsd_Cr = np.sqrt(np.sum((Cr - mean_Cr - (Cr - mean_Cr))**2) / N)
    rmsd_Cf = np.sqrt(np.sum((Cf - mean_Cf - (Cr - mean_Cr))**2) / N)
    
    # Compute Correlation Coefficient
    corr_Cr = np.sum((Cr - mean_Cr) * (Cr - mean_Cr)) / (N * std_Cr**2)
    corr_Cf = np.sum((Cf - mean_Cf) * (Cr - mean_Cr)) / (N * std_Cf * std_Cr)
    
    # Assemble the result in a 4x2 array (for each statistic)
    STATM = np.array([[mean_Cr, mean_Cf],
                      [std_Cr, std_Cf],
                      [rmsd_Cr, rmsd_Cf],
                      [corr_Cr, corr_Cf]])

    return STATM