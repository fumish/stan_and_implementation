functions{
  matrix sigmoid_matrix(matrix x){
    return (1 ./ (1 + exp(-x)));
  }
  row_vector sigmoid_vector(row_vector x){
    return (1 ./ (1 + exp(-x)));
  }
  
  vector sigmoidal_neural_network(matrix x, matrix b, row_vector a){
    return sigmoid_vector(a*sigmoid_matrix(b*x'))';
  }
}

data{
  int<lower=1> n; //num of sample
  int<lower=1> M; //dim of input
  vector[n] y; //sample for output
  matrix[n,M] x; //sample for intput
  int<lower=0> H; //num of hidden
  
  real sigma; //sd of learning model
}
transformed data{
  real<lower=0> lambda;
  lambda <- 4;
}

parameters{
  matrix[H,M] b; //weights between input and hidden
  row_vector[H] a; //weights between hidden and output
}

model{
  // real squared_error;
  
  //for wide prior
  for(i in 1:H){
    for(j in 1:M){
      b[i,j] ~ normal(0,lambda);
    }
  }
  for(j in 1:H){
      a[j] ~ normal(0,lambda);
  }
  
  y ~ normal(sigmoidal_neural_network(x, b, a), sigma);
  
//   squared_error <- 0;
//   for(i in 1:n){
//     squared_error <- squared_error + normal_log(y[i], sigmoidal_neural_network(x[i]', b, a), sigma);
//   }
//   increment_log_prob(squared_error);
}