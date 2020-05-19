/**
*   This is code to evalue LS-SVR (Least Squares Support Vector Regression).
*   Currently this solely is suitable for one dimensional output. Will be updated accordingly.
*   This code is more memorry intensive but faster for large sets.
**/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstring>
#include <sstream>
#include "mpi.h"
#include <stdlib.h>
#define TWO_PI 6.2831853071795864769252866  

#define PI  3.14159265358979323846 
using namespace std;

int Thresh(double * n,double cutoff, int N)
{
    double err = 0;
    for(int i = 0; i < N; i++) {
        err += n[i]*n[i];
    }
    if(sqrt(err) < cutoff) {return 0;}
    else {return 1;}
}

int conjG(double *r, double *x, double ** A, int N, int cut, double gamma, int iT) {
    double a = 0.0;
    double num = 0.0;
    double denom = 0.0;
    double beta = 0.0;
    int iter = 1; 
    int outer_N = 0;
    int inner_N = 0;
    int count = 0;
    double cutoff = 1e-10;
    double g = 1.0/gamma;
    double p[N];
    double Ap[N];
    double xn[N];
    double rn[N];
    for(int i = 0; i < N; i++) {
        p[i] = 0.0;
        Ap[i] = 0.0;
        xn[i] = x[i];
        rn[i] = r[i];
    }

    for(int i = 0; i < N; i++) {
        p[i] = r[i]/(1.0+g);
        num += r[i]*p[i];
    }
    int b = 1;
    while(iter < cut) {
        count = 0;
        inner_N = 0;
        outer_N = 0;
        for(int i = 0; i < N; i++) {
            Ap[i] = (1.0+g)*p[i];
            for(int j = i+1; j < N; j++) {
                Ap[i] += p[j]*A[outer_N][inner_N];
                if(inner_N < iT-1) {inner_N++;}
                else{inner_N = 0; outer_N++;}
            } 
        } 

        count = 0;
        inner_N = 0;
        outer_N = 0; 

        for(int i = 0; i < N-1; i++) {
            for(int j = i+1; j < N; j++) {
                Ap[j] += p[i]*A[outer_N][inner_N];
                if(inner_N < iT-1) {inner_N++;}
                else{inner_N = 0; outer_N++;}
            }           
        } 
        
        denom = 0.0;
        for(int i = 0; i < N; i++) {
            denom += Ap[i]*p[i]; 
        }      
        a = num/denom;
        
        for(int i = 0; i < N; i++) {
            xn[i] = xn[i] + a*p[i];
            rn[i] = rn[i] - a*Ap[i];
        }
        
        b = Thresh(rn,cutoff,N);
        if(b == 0) {break;} 
        denom = num;
        num = 0.0;
        for(int i = 0; i < N; i++) {
            num += rn[i]*rn[i]/(1.0+g);
        }
        beta = num/denom;
        
        for(int i = 0; i < N; i++) {
            p[i] = rn[i]/(1.0+g) + beta*p[i];
        }
        
        for(int i = 0; i < N; i++) {
            for(int j = 0; j< N; j++) {
                Ap[i] = 0.0;
            }
            
        }
        iter++;
    }
    for(int i = 0; i < N; i++) {
        x[i] = xn[i];
        r[i] = rn[i];
    }

    return iter;
    
}


int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);
    int my_rank;
    int np;
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    MPI_Status status;

    
    int N=atoi(argv[1]); // Number of training points
    
    int d=atoi(argv[2]);  // Dimension of input
    double gamma = atof(argv[3]);
    double sigma2= atof(argv[4]); 
    
    
    double L2;
    
    int i1, i2;
    int tcount = 0;
    int vcount = 0;
    int tcount_o = 0;
    int tcount_i = 0;
    int kfold = np;
    int V = N/kfold;
    int T = N-V;
    int M = T*(T-1)/2;

    int outer_T = T/10;
    int inner_T = 5*(T-1);   
    int cut = atoi(argv[5]);


    double **x_train;
    double *y_train;
    double **x_val;
    double *y_val;
    double ** K_train;
    K_train = new double* [outer_T];

    for(int i = 0; i < outer_T; i++) {
        K_train[i] = new double [inner_T];
    }


    double *K_val;
    K_val = new double [V*T];

    double* rv;
    double* xv;
    double* rn;
    double* xn;

    double* alpha;
    double* y_new;

    double sum, b, x2;
    double xtemp,xmin;
    






    alpha = new double [T];
    rn = new double [T];
    xn = new double [T];

    rv = new double [T];
    xv = new double [T];

    y_new = new double [V];
    y_val = new double [V];
    y_train = new double [T];


    x_val = new double* [V];
    for(int i = 0; i < V; i++) {
        x_val[i] = new double [d];
    }  

    x_train = new double* [T];
    for(int i = 0; i < T; i++) {
        x_train[i] = new double [d];
    } 

    for(int i = 0; i < T; i++) {
        for(int j = 0; j < d; j++) {
            x_train[i][j] = 0.0;
        }
        y_train[i] = 0.0;
    }
    for(int i = 0; i < V; i++) {
        for(int j = 0; j < d; j++) {
            x_val[i][j] = 0.0;
        }
        y_val[i] = 0.0;
        y_new[i] = 0.0;
    }

    int gmes = 0;
   
   
    ofstream screen;
    string direct;
    stringstream d1;
    stringstream d2;
    int ttemp = N/100;
    d1 << setw(4) << setfill('0') << ttemp;
    string dd1 = d1.str();
    int ztemp = my_rank+1;
    d1 << ttemp;
    d2 << ztemp;
    string dd2 = d2.str();
    direct = "Progress_" + dd1 + "_" + dd2 + ".txt";
    char *fd = new char[direct.length()+1];
    strcpy(fd,direct.c_str()); 
 
    screen.open(fd,ios::app);
   // screen << "*****BEGINNING CROSS VALIDATION******" << endl;
   // screen << "Size of training set: " << N << endl;
   // screen << "K-fold validation number: " << kfold << endl;
   // screen << "Processor " << my_rank+1 << endl;  
         
    if(my_rank == 0) {


        string idirect;
        idirect = "../Full_Data/training_" + dd1 + ".dat";
        char* id = new char[idirect.length()+1];
        strcpy(id,idirect.c_str()); 


        int index = 0;
        int cnt = 0;

                
        ifstream in;

        in.open(id);
        

        
        double** x_in;
        double* y_in;
        
        x_in = new double* [N];
        for(int i = 0; i < N; i++) {
            x_in[i] = new double [d];
        }
        


        y_in = new double [N];

              
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < d; j++) {
                in >> x_in[i][j];
            }
            in  >> y_in[i];
        }
        
                
        for(int z = 1; z < np; z++) {

            for(int i = 0; i < V; i++) {
                for(int j = 0; j < d; j++) {
                    MPI_Send ( &(x_in[z*V+i][j]), 1, MPI_DOUBLE, z, V*i+j, MPI_COMM_WORLD );                    
                }
                MPI_Send (&(y_in[z*V+i]), 1, MPI_DOUBLE, z, 0, MPI_COMM_WORLD );
                   
            }    
                         
        }
        
        for(int z = 1; z < np; z++ ){
            index = 0;
            cnt = 0;
            while(cnt < T) {
                if(index == z*V) {
                    index = (z+1)*V;
                }
                for(int j = 0; j < d; j++) {
                    MPI_Send ( &(x_in[index][j]), 1, MPI_DOUBLE, z, j, MPI_COMM_WORLD );                    
                }
                MPI_Send (&(y_in[index]), 1, MPI_DOUBLE, z, 0, MPI_COMM_WORLD );
                index++;
                cnt++;
            }
        }
        
        for(int i = 0; i < V; i++) {
            for(int j = 0; j < d; j++) {
                x_val[i][j] = x_in[i][j];
            }
            y_val[i] = y_in[i];
        }   
        
        for(int i = 0; i < T; i++) {
            for(int j = 0; j < d; j++) {
                x_train[i][j] = x_in[V+i][j];
            }   
            y_train[i] = y_in[V+i];
        }
        
        for(int i = 0; i < N; i++) {
            delete [] x_in[i];
        }
        delete [] x_in;
        delete [] y_in;
        
        in.close();
        
    }
    else { 
    
        for(int i = 0; i < V; i++) {
            for(int j = 0; j < d; j++) {
                MPI_Recv(&(x_val[i][j]), 1, MPI_DOUBLE,0,V*i+j,MPI_COMM_WORLD,&status);
            }
            MPI_Recv(&(y_val[i]), 1, MPI_DOUBLE, 0,0,MPI_COMM_WORLD,&status);            
        }   
            
        for(int i = 0; i < T; i++) {
            for(int j = 0; j < d; j++) {
                MPI_Recv(&(x_train[i][j]), 1, MPI_DOUBLE,0,j,MPI_COMM_WORLD,&status);
            }           
            MPI_Recv(&(y_train[i]), 1, MPI_DOUBLE, 0,0,MPI_COMM_WORLD,&status);            
        }   	
	
    }
        

    tcount = 0;
    vcount = 0;



    vcount = 0;
    double xdiff = 0.0;
    for(int i = 0; i < T; i++) {
        
        for(int j = i+1; j < T; j++) {
            x2 = 0.0;
            for(int k = 0; k < d; k++) {
                xdiff = x_train[i][k]-x_train[j][k];
                if(xdiff > PI) {xdiff -= TWO_PI;}
                if(xdiff < -PI) {xdiff += TWO_PI;}
                x2 += xdiff*xdiff;
            }
           
           K_train[tcount_o][tcount_i] = exp(-x2/sigma2/2.0); 
           if(tcount_i < inner_T-1) {tcount_i++;}
           else{tcount_i = 0; tcount_o++;}
                     
           
        }
    
        for(int j = 0; j < V; j++) {
            x2 = 0.0;
            for(int k = 0; k < d; k++) {
                xdiff = x_train[i][k]-x_val[j][k];
                if(xdiff > PI) {xdiff -= TWO_PI;}
                if(xdiff < -PI) {xdiff += TWO_PI;}
                x2 += xdiff*xdiff;
            }
           K_val[vcount] = exp(-x2/sigma2/2.0);                      
           vcount++; 
        }         
                         
    }
    
    for(int i = 0; i < V; i++) {
        delete [] x_val[i];
    }
    delete [] x_val;
    
    for(int i = 0; i < T; i++) {
        delete [] x_train[i];
    }
    delete [] x_train;    

    

       

       
    screen << gamma << " " << sigma2 << " ";     
    for(int i = 0; i < T; i++) {
        xn[i] = 0.0;
        rn[i] = 0.0;
        rv[i] = 0.0;
        xv[i] = 0.0;
        
        alpha[i] = 0.0;
        
    }
    for(int j = 0; j < V; j++) {
        y_new[j] = 0.0;
     }
     
    for(int i = 0; i < T; i++) {
        rv[i] = y_train[i];
        rn[i] = 1.0;
    }   
    

    sum = 0.0;
    
    
    i1 = conjG(rn,xn,K_train,T,cut,gamma,inner_T); 
    i2 = conjG(rv,xv,K_train,T,cut,gamma,inner_T);
    for(int i = 0; i < T; i++) {       
        sum += xn[i];
    }
    
    b = 0.0;

    for(int i =0; i < T; i++) {
        b += xn[i]*y_train[i]/sum;
        
    }
       
    for(int i = 0; i < T; i++) {
        alpha[i] = xv[i] - xn[i]*b;
    }

    x2 = 0.0;

    for(int j = 0; j < V; j++) {
        for(int i = 0; i < T; i++) {
            y_new[j] += K_val[i*V+j]*alpha[i];
        }
    }
    L2 = 0.0;
    
    for(int i = 0 ; i < V; i++) {
        L2 +=  (y_val[i]-y_new[i]-b)*(y_val[i]-y_new[i]-b)/((double)V);
    }
    L2 = sqrt(L2);
    screen << i1 << "   " << i2 << "    " << L2 << endl;            

       
        
    delete [] y_new;
    delete [] y_val;        
    delete [] y_train;
    
    
    delete [] rn;
    delete [] xn;
    delete [] rv;
    delete [] xv;        
    delete [] alpha;

    
   
    delete [] K_train;
    delete [] K_val;
    
	    
	screen.close();

    
    MPI_Finalize();
    
    
    
	    return 0;
}


