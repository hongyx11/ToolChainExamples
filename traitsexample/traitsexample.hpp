
#pragma once

// header include starts 
// header include ends

// impl include starts
// impl include ends


namespace combblas {

template<class NT1, class NT2,class Enabled= void>
struct promote_traits{};
    template<>
struct promote_traits<double,float>{
        typedef double T_promoted;
    };
    template<>
    struct promote_traits<float,float>{
        typedef float T_promoted;
    };

// Template function that uses promote_traits to determine its return type.
template<typename NT1, typename NT2>
typename promote_traits<NT1, NT2>::T_promoted promote_add(NT1 a, NT2 b) {
    // Here, you could do any operation that returns the promoted type.
    // For example, simply add the two numbers.
    return a + b;
}

} // namespace PROJNAME ends
