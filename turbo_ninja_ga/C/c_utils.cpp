#include <boost/python.hpp>
#include <stdlib.h>

namespace py = boost::python;


py::list choose_mutation_posns(double length){    
    py::list mutation_posns;
    for(int i=0;i<length;++i){
        if ((float) rand()/RAND_MAX < 1.0/length){
            mutation_posns.append(i);
        }
    }
    return mutation_posns;
}

double list_sum(py::list l){
    double result = 0.0;
    for(int i=0;i<len(l);++i){
        result = result + py::extract<double>(l[i]);
    }
    return result;
}

py::list uniform_fuck(py::list A, py::list B){
    py::list child;
    for(int i=0;i<len(A); ++i){
        if ((float) rand()/RAND_MAX < 0.5){
            child.append(A[i]);
        }
        else{
            child.append(B[i]);        
        }
    }
    return child;
}

BOOST_PYTHON_MODULE(c_utils)
{
    using namespace boost::python;
    def("choose_mutation_posns", choose_mutation_posns);
    def("list_sum", list_sum);
    def("uniform_fuck", uniform_fuck);
}