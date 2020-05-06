# Background_Recovery_by_decomposition_into_Low_Rank_and_Sparse_Components

The goal of this project is to analyze and extract the background information from a timelapse/surveillance video. To solve the dynamic segmentation problems that are encountered in imaging, we can utilize many sparse representation techniques and tools. We utilize the sparsity of a signal to solve an underdetermined system of linear equations. The method we will be utilizing is called RPCA (Robust Principal Component Analysis)​[1]​ which is a sparse and low rank matrix decomposition problem. We will assume a signal Y ∈ ℝ​m×n ​ that contains static and dynamic components (In our instance it is L= Background; S= Moving objects e.g. cars). Next, we will estimate the values of L and S from the observation Y by solving the following optimization problem: 
 ● Minimize ∥L∥∗+λ∥S∥​1 subject to : L+S=Y  ..(1) 
 
where ║ · ║∗ denotes the nuclear norm. 
 
The equation can then be modified to include a second constraint that will allow us to choose which frequencies will be allowed in S. Thus, the new optimization problem will be: 
 ● Minimize ∥L∥∗+λ∥SF​T​∥​1 subject to : L +S=Y, SF​T​Γ=0 ..(2) 
 where Γ ∈ ℝ​n×n​ is a diagonal matrix of 1’s and 0’s. 
 We’ll be using the Alternating Direction Method of Multipliers for solving the constrained optimization problem. 
 
 
References:  [1] McLean JP, Ling Y, Hendon CP. Frequency-constrained robust principal component analysis:  a sparse representation approach to segmentation of dynamic features in optical coherence  tomography imaging. ​Opt Express​. 2017;25(21):25819–25830. doi:10.1364/OE.25.025819 
