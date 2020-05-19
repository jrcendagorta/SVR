/**
*   This is code to evalue LS-SVR (Least Squares Support Vector Regression).
*   Currently this solely is suitable for one dimensional output. Will be updated accordingly.
**/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <string>
#include <stdlib.h>
#include <sstream>


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

void conjG(double *r, double *x, double ** A, int N, double gamma, int iT) {
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
    int cut = 2000000;
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
}

int main(int argc, char* argv[]) {
    int N; // Number of training points
    
    int d;  // Dimension of input
    double gamma = atof(argv[3]);
    double sigma2 = atof(argv[4]); 
        
    
    N = atoi(argv[1]);
    d = atoi(argv[2]);
    
    int V = atoi(argv[5]);
    int T = N;
    
    ifstream in, val_in;
    
    
    
    ofstream out, val_out, data_out;
    string direct,vdirect,ddirect;
    stringstream d1;
    int ttemp = N/100;
    d1 << setw(4) << setfill('0') << ttemp;
    string dd1 = d1.str();
    direct = "Model_" + dd1 + ".txt";
    char *fd = new char[direct.length()+1];
    strcpy(fd,direct.c_str()); 
    vdirect = "Validation_" + dd1 + ".txt";
    char* vd = new char[vdirect.length()+1];
    strcpy(vd,vdirect.c_str()); 
    ddirect = "Difference_" + dd1 + ".txt";
    char* dd = new char[ddirect.length()+1];
    strcpy(dd,ddirect.c_str()); 

    string idirect;
    idirect = "../Full_Data/training_" + dd1 + ".dat";
    char* id = new char[idirect.length()+1];
    strcpy(id,idirect.c_str()); 
    in.open(id);
    val_in.open("../Full_Data/test_set.dat");
    out.open(fd);
    val_out.open(vd);
    data_out.open(dd);
    

    double** x_val;
    double* y_val;
    
    double** x_train;
    double* y_train;
        

    
    double* rv;
    double* xv;
    double* rn;
    double* xn;

    double* alpha;
    double* y_new;
    
    int index = 0;
    
    x_val = new double* [V];
    for(int i = 0; i < V; i++) {
        x_val[i] = new double [d];
    }  
    
    x_train = new double* [T];
    for(int i = 0; i < T; i++) {
        x_train[i] = new double [d];
    }  

    y_new = new double [V];
    y_val = new double [V];
    y_train = new double [T];
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < d; j++) {
            in >> x_train[i][j];
        }
        in  >> y_train[i];
    }
    
    for(int i = 0; i < V; i++) {
        for(int j = 0; j < d; j++) {
            val_in >> x_val[i][j];
        }
        val_in >> y_val[i];
        y_new[i] = 0.0;
    }
    
    alpha = new double [T];
    rn = new double [T];
    xn = new double [T];
    
    rv = new double [T];
    xv = new double [T];
    
    int cnt = 0;
    
    double sum, b, x2, L2, L2_sum;
    
    double **K_train;
    int outer_T = T/10;
    int inner_T = 5*(T-1);
    K_train = new double* [outer_T];

    for(int i = 0; i < outer_T; i++) {
        K_train[i] = new double [inner_T];
    }

    double *K_val;
    int V_new = ((int)((double)T)*((double)V));
    K_val = new double [V_new];
    int vcount = 0;
    int tcount_o = 0;
    int tcount_i = 0;
    double xmin, xtemp;
    int quad;
    
    L2_sum = 0.0;

  
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

             
    for(int i = 0; i < T; i++) {
            rn[i] = 0.0;
            xn[i] = 0.0;
            
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
    cout << "STARTING CONG 1...." << endl;
    conjG(rn,xn,K_train,T,gamma,inner_T);
    cout << "FINISHED CONG 1...." << endl;
    cout << "STARTING CONG 2...." << endl;
    conjG(rv,xv,K_train,T,gamma,inner_T);
    cout << "FINISHED CONG 2...." << endl;   
        
    for(int i = 0; i < T; i++) {
        sum += xn[i];
    }
            
    b = 0.0;

    for(int i =0; i < T; i++) {
        b += xn[i]*y_train[i]/sum;
        
    }
    
    out << b << "   " << sigma2 << " " << gamma << endl;
    
    for(int i = 0; i < T; i++) {
        alpha[i] = xv[i] - xn[i]*b;
        out << i+1 <<"	" << setprecision(7)<< alpha[i]  << "   " ;
        for(int j = 0; j < d; j++) {
            out << x_train[i][j] << "   ";
        }
        out <<  endl;
    }

    x2 = 0.0;
    cout << "VALIDATING...." << endl;
    for(int j = 0; j < V; j++) {
        for(int i = 0; i < T; i++) {
            y_new[j] += K_val[V*i+j]*alpha[i];           
        }
    }
  
    
    L2 = 0.0;
    for(int i = 0 ; i < V; i++) {
        L2 =  (y_val[i]-y_new[i]-b)*(y_val[i]-y_new[i]-b);
        L2_sum += L2;
        val_out << i << "  " << sqrt(L2_sum/((double)i+1)) << endl;
        for(int j = 0; j < d; j++) {
            data_out << x_val[i][j] << "    ";
        }
        data_out << y_val[i] << "   " << y_new[i]+b << endl;
    }
    

    

    delete [] y_val;
    delete [] y_new;
    delete [] y_train;
    delete [] rn;
    delete [] xn;
    
    delete [] alpha;

    delete [] rv;
    delete [] xv;
    for(int i = 0; i < outer_T; i++) {
        delete [] K_train[i];
    }
    delete [] K_train;
    delete [] K_val;
    
    
    for(int i = 0; i < V; i++) {
        delete [] x_val[i];
    }
    delete [] x_val;
    
    for(int i = 0; i < T; i++) {
        delete [] x_train[i];
    }
    delete [] x_train;
    
    out.close();
    in.close();
    val_in.close();
    val_out.close();
    data_out.close();
    
}
