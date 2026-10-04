#include "local.h"

#include <cmath>


// PDE: -\Delta u + \sigma u = f
//  BC: du/dn = 0

const double sigma = 100;

double f(double x, double y) {
    double s = 0.25;
    double dx = x - 0.4;
    double dy = y - 0.3;

    return 1000.0 * std::exp(
        -(dx * dx + dy * dy) / (2.0 * s * s)
    );
}

/***********************************************/

struct LocalMatrix22 {
    double el[2][2];
};

struct LocalVector2 {
    double el[2];
};

const double grads[3][2]{
    {-1, -1},
    {1, 0},
    {0, 1}
};

LocalMatrix22 jacobian(Triangle t) {
    return {
        {
            {t.verts[1]->x - t.verts[0]->x, t.verts[2]->x - t.verts[0]->x},
            {t.verts[1]->y - t.verts[0]->y, t.verts[2]->y - t.verts[0]->y}
        }
    };
}

double det(LocalMatrix22 jac) {
    return jac.el[0][0] * jac.el[1][1] - jac.el[1][0] * jac.el[0][1];
}

LocalMatrix22 inv(LocalMatrix22 m) {
    double d = det(m);
    return {
        {
            {m.el[1][1] / d, -m.el[0][1] / d},
            {-m.el[1][0] / d, m.el[0][0] / d}
        }
    };
}

LocalMatrix22 trans(LocalMatrix22 m) {
    return {
        {
            {m.el[0][0], m.el[1][0]},
            {m.el[0][1], m.el[1][1]}
        }
    };
}

LocalMatrix22 mm(LocalMatrix22 a, LocalMatrix22 b) {
    LocalMatrix22 res;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            double s = 0;
            for (int k = 0; k < 2; k++) {
                s += a.el[i][k] * b.el[k][j];
            }
            res.el[i][j] = s;
        }
    }

    return res;
}

LocalVector2 mv(LocalMatrix22 a, LocalVector2 v) {
    LocalVector2 res;
    for (int i = 0; i < 2; i++) {
        double s = 0;
        for (int k = 0; k < 2; k++) {
            s += a.el[i][k] * v.el[k];
        }
        res.el[i] = s;
    }

    return res;
}

double dot(LocalVector2 v, LocalVector2 w) {
    double s = 0;
    for (int k = 0; k < 2; k++) {
        s += v.el[k] * w.el[k];
    }
    return s;
}

double form(int i, int j, double d, LocalMatrix22 matr) {
    LocalVector2 g1{
        {grads[i][0], grads[i][1]}
    };

    LocalVector2 g2{
        {grads[j][0], grads[j][1]}
    };

    return 0.5 * d * dot(mv(matr, g1), g2) + d * (i == j ? sigma / 12.0 : sigma / 24.0);
}

LocalMatrix get_local_matrix(Triangle t) {
    LocalMatrix22 jac = jacobian(t);
    double d = std::abs(det(jac));
    LocalMatrix22 matr = inv(mm(trans(jac), jac));
    LocalMatrix res;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            res.el[i][j] = form(i, j, d, matr);
        }
    }
    return res;
}

LocalVector get_local_vector(Triangle t) {
    const double area = 0.5 * std::abs(det(jacobian(t)));
    double f_avg = 0;
    for (int i = 0; i < 3; i++) {
        f_avg += f(t.verts[i]->x, t.verts[i]->y);
    }
    f_avg /= 3;

    double integral = f_avg * area / 3;

    return {
        {integral, integral, integral}
    };
}
