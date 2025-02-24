//
// Created by Yuxi Hong on 2/24/25.
//
#include "traitsexample.hpp"
#include "another.hpp"
namespace combblas {

    template typename promote_traits<double,float>::T_promoted promote_min<double, float>(double, float);

    template typename promote_traits<float,float>::T_promoted promote_add<float, float>(float, float);

}