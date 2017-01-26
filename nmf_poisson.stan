data{
  int<lower=1> n; //number of sample
  int<lower=1> N; //row dimension of input
  int<lower=1> H; //hidden dimension
  int<lower=1> M; //column dimension of input
  
  int x[n,N,M]; //matrix to be decomposed by A and B
  
  real<lower=0> alpha; //hyperparameter for gamma dist
  real<lower=0> beta; //hyperparameter for gamma dist
}

parameters{
  matrix<lower=0>[N, H] A; //non-neg constraint
  matrix<lower=0>[H, M] B; //non-neg constraint
}


model{
  matrix[N,M] AB;
  AB = A*B;
  
  //assuming prior is gamma distribution
  for(i in 1:N){
    for(j in 1:H){
      A[i,j] ~ gamma(alpha,beta);
    }
  }
  for(i in 1:H){
    for(j in 1:M){
      B[i,j] ~ gamma(alpha,beta);
    }
  }

  for(j in 1:N){
    for(k in 1:M){
      target += poisson_lpmf(x[,j,k] | AB[j,k]);
    }
  }
}