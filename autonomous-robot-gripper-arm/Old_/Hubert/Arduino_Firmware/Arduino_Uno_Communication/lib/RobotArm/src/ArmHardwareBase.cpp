#include "ArmHardwareBase.h"
#include <vector>

ArmHardwareBase::ArmHardwareBase(const ArmConfig& config) :
    arm_config(config) 
                      // like doing self.arm_config = config in python
{
    // no memory allocation here
}


LinRegResults ArmHardwareBase::lin_reg(const std::vector<double> y_vec, const std::vector<double>  x_vec) 

{
const size_t n = x_vec.size();
double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x_squared = 0.0;

for (size_t i = 0; i < n; ++i) 
{
    sum_x += x_vec[i];
    sum_y += y_vec[i];
    sum_xy += x_vec[i] * y_vec[i];
    sum_x_squared += x_vec[i] * x_vec[i];
}

double denominator = n * sum_x_squared - sum_x * sum_x;
double slope = (n * sum_xy - sum_x * sum_y) / denominator;


double intercept = (sum_y - slope * sum_x) / n;

return {slope, intercept};
}
        

