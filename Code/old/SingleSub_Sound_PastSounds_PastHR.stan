// This model fits single sessions (no hierarchy)
// components:  learning rate for heart rate, learning rate for sounds, regression weights for sound, heart rate, past sounds
// paper: Symmonds 2011


// To fit single sessions by themselves, let's give them a prior
data{
  int ntr;          // number of total data points
  real listenBPM[ntr];
  real responseBPM[ntr];
  int Decision.num[ntr];  // choice on each trial (1 for left, 0 for right)

}

parameters{
  // These are for individual sessions
  real b_soundThis;
  real b_soundPast;
  real b_heart;
  real<lower=0,upper=1> lr_soundPast;
  real<lower=0,upper=1> lr_response;
}


model{
  real belief_soundPast[ntr+1];
  real belief_heart[ntr+1];
  real belief_heartPast[ntr+1];
  real decVal[ntr];
  
  // Priors
  b_soundThis ~ normal(0,3);
  b_soundPast ~ normal(0,3);
  b_heart     ~ normal(0,3); 
  
  // Learning
  for (itr in 1:ntr){
    if (itr==1){
      belief_soundPast[itr] = responseBPM[itr];
      belief_heartPast[itr] = listenBPM[itr];
    }
    // for sound: really project forward like normal RL
    // for heart:later need to shift it back again by one trial
    belief_soundPast[itr+1] = belief_soundPast[itr] + lr_soundPast*(responseBPM[itr]-belief_soundPast[itr])
    belief_heartPast[itr+1] = belief_heartPast[itr] + lr_heartPast*(listenBPM[itr]-belief_heartPast[itr])
    
    belief_heart[itr] = belief_heartPast[itr+1];
    
    decVal[itr] = b_soundThis*responseBPM[itr] + b_soundPast*belief_soundPast[itr] + b_heart*belief_heart[itr];
  }
  
  // choice
  decVal ~ bernoulli_logit(decVal);
    
    
  }
}
  