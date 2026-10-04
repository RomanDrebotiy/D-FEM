#pragma once

#include "../mesh/geometry.h"

struct LocalMatrix {
    double el[3][3];
};

struct LocalVector {
    double el[3];
};

LocalMatrix get_local_matrix(Triangle t);

LocalVector get_local_vector(Triangle t);