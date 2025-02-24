#include "another.hpp"
#include "traitsexample.hpp"

int main(){
    double d1 = 10.2343;
    float f1 = 10.342;
    auto res = combblas::promote_add<double,float>(d1,f1);
    auto res2 = combblas::promote_min(d1,f1);
    auto res3 = combblas::promote_add(f1,f1);
return 0;
}