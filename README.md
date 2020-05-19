# Least Squares Support Vector Regression for Peptides

This project was part of a larger collaborative effort in using machine learning models
to determine the most efficient and accurate method in predciting the free energy surfaces
of a variety of oligopeptides. 

Currently, the LS-SVR method is broken into two different code structures. The LS-SVR alogrithm used
can be found in the Johan Suykens perspective: https://www.sciencedirect.com/science/article/abs/pii/S0947358001711521

1) LSSVR_kfoldCV_MPI.cpp
	This file will perform a k-fold crossvalidation for a given set of hyperparameters (gamma and sigma^2).
	The value k-fold will depend on the the number of n processors used (n=k). The program is written in parallel
	to reduce the computational time for cross validation. A conjugate gradient method is used to determine the 
	model weights. Currently, the program will accept a single set of hyperparameters and output the resulting error from cross
	validation. Further implementations will want to introduce various hyperparameter searching methods within the code.
	To compile:

		mpic++ LSSVR_kfoldCV_MPI.cpp -o LSSVR_kfold-mpi

	To effectively use the program a sample script is shown below that can be used in any job queueing system:

 		sigma=(0.6 0.7 0.8)
		gamma=(5 10 20 50)
		for i in "${sigma[@]}"
		do
    		for j in "${gamma[@]}"
    			do
        			mpirun  -np 10 ./a.out 50000 6 $j $i 5000
    			done
		done
	where the terms in the command argument are the number of training points, the number of feautres (input dimension), 
	value of gamma, sigma, conjugate gradient cutoff iterations.

	The program will output k files, one for each processor (Progress_XXXX_ID.txt where XXXX is the number of training points/10 and 
	is the rank of processor) that has the hyperparameters followed by the number of conjugate gradient
	steps for the matrix inversions needed for the method, followed by the loss (l2) error. From this, you can average (post-process)
	over all folds and ensure that the hyperparameters obtained converged within the threshold of the number of conjugate gradient 
	iterations. 

	Future iterations of this program will do all the post-processing prior to program termination. 

2) LSSVR_Model.cpp 
	After determining the appropriate hyperparameters, this code will perform the model training and generation.
	It will print out the model parameters and weights in the output file (Model_XXXX.txt). The program is not in 
	parallel and is compiled:

		g++ LSSVR_Model.cpp -o LSSVR_Model
	
	The program also outputs two additional files Validation_XXX.txt which keeps track of the running average of the
	error as a function of the number of test points and Difference_XXXX.txt which prints all the training points, 
	the true target value and the predicted target value.  
