functions{
  real squared_error(matrix x, matrix A, 
  matrix B){
    return trace((x-A*B)*(x-A*B)');
  }
}

data{
  int<lower=1> n; //number of sample
  int<lower=1> N; //row dimension of input
  int<lower=1> H; //hidden dimension
  int<lower=1> M; //column dimension of input
  
  matrix[N,M] x[n];
  
  real<lower=0> learning_sigma;
  
  real<lower=0> lambda;
}

parameters{
  matrix<lower=0>[N, H] A;
  matrix<lower=0>[H, M] B;
}


model{
  // for ridge prior
  for(i in 1:N){
    for(j in 1:H){
      A[i,j] ~ normal(0,lambda);
    }
  }
  for(i in 1:H){
    for(j in 1:M){
      B[i,j] ~ normal(0,lambda);
    }
  }
// 
//   //for lasso prior
//   for(i in 1:N){
//     for(j in 1:H){
//       A[i,j] ~ double_exponential(0,lambda);
//     }
//   }
//   for(i in 1:H){
//     for(j in 1:M){
//       B[i,j] ~ double_exponential(0,lambda);
//     }
//   }

  for(i in 1:n){
    target += -squared_error(x[i],A,B)/(2*learning_sigma*learning_sigma);
  }
}