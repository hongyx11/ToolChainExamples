//
// Created by Yuxi Hong on 2/24/25.
//

#include "traitsexample.hpp"

namespace combblas {





    template typename promote_traits<double,float>::T_promoted promote_add<double, float>(double, float);

}