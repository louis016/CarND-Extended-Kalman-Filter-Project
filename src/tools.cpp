#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  
  // Calculate the RMSE here.
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  VectorXd residual;

  // accumulate the residuals
  for (int i = 0; i < estimations.size(); ++i) {
     residual = estimations[i] - ground_truth[i];
     residual = residual.array() * residual.array();
     rmse += residual; 
  }
  
  // caculate the average
  rmse = rmse / estimations.size();
  
  // caculate the rmse
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  // Calculate a Jacobian here.
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  MatrixXd Hj(3,4);
  float div1 = px * px + py * py;

  if(fabs(div1)< 0.0001){
    px += 0.001;
    py += 0.001;
  }

  float div2 = sqrt(div1);
  float div3 = pow(div1,3/2);
 
  Hj << px / div2, py / div2, 0, 0,
        -py / div1, px / div1, 0, 0,
        py * (vx * py - vy * px) / div3, px * (vy * px - vx * py) / div3, px / div2, py / div2;

  return Hj;
}
