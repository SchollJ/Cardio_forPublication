// average of listen beeps is learned with a learning rate


data {
  int ntr;
  real listenBPM[ntr];
  real responseBPM[ntr];
  
  }

// The parameters accepted by the model
parameters {
  real<lower=0> invTemp;
  real<lower=0,upper=1>learning_rate;
}

model {
  real beliefAvg[ntr];
  for (itr in 1:ntr){
    beliefAvg[itr+1] = beliefAvg[itr] + learning_rate*(responseBPM[itr]-beliefAvg[itr]);
    
    
  }
}
