#include <fstream>
#include <iostream>
#include <cmath>
#include "vector.h"
#include "eigen-3.4.0/Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// --------------- Global constants ---------------

const int Lx = 720;
const int Ly = 540;
const int N = 1;   // Number of bodies
const int Nm = 60; // Number of markers to discretize the body
const int Ls = 3;  // Length of support for interpolation

const int Q = 9;
const double tau = 0.54;
const double nu = 1.0 / 3 * (tau - 0.5);
const double Utau = 1.0 / tau;
const double UmUtau = 1 - Utau;
const double ThreeUmU2tau = 3 * (1 - 1 / (2 * tau));

// --------------- Class & global functions declaration ---------------

class Body;
class LatticeBoltzmann;
class Engine;

double delta(double r);
double mollifier(double r1, double r2);

// --------------- Class implementation ---------------

class Body
{ // Flat plate
private:
public:
    vector3D r[Nm], V[Nm], F[Nm];
    vector3D Fib[Nm];
    VectorXd eps;
    double h, rhom, Vm, ds; // Height, density, volume, segment length
    void Start(double x0, double y0, double Vx0, double Vy0, double rhom0, double h0);
    void GetEps(void);
    void ResetForce(void)
    {
        for (int i = 0; i < Nm; i++)
        {
            Fib[i].load(0.0, 0.0, 0);
            F[i].load(0.0, 0.0, 0);
        }
    };
    void AddForce(vector3D &dF)
    {
        for (int i = 0; i < Nm; i++)
            F[i] += dF;
    };
    void Move_r(double dt, double coeff)
    {
        for (int i = 0; i < Nm; i++)
            r[i] += V[i] * (coeff * dt);
    };
    void Move_V(double dt, double coeff)
    {
        for (int i = 0; i < Nm; i++)
            V[i] += F[i] * (coeff * dt);
    };
    double Getx(int i) { return r[i].x(); };
    double Gety(int i) { return r[i].y(); };
    void Print(int t);
    friend class Engine;
    friend class Lattice;
};

class Engine
{
private:
public:
    void GetTwoBodyForces(Body &B1, Body &B2);
    void GetLatticeForces(double gx, double gy, Body &B, Body *Bodies, LatticeBoltzmann &Lattice);
    void GetAllForces(double gx, double gy, Body *Bodies, LatticeBoltzmann &Lattice);
};

class LatticeBoltzmann
{
private:
    double w[Q];      // Weights
    int Vx[Q], Vy[Q]; // Velocity vectors of D2Q9
    double *f, *fnew; // Distribution functions
public:
    LatticeBoltzmann(void);
    ~LatticeBoltzmann(void);
    int n(int ix, int iy, int i) { return (ix * Ly + iy) * Q + i; };

    // --------------- Macroscopic fields ---------------

    double rho(int ix, int iy, bool UseNew);
    double Jx(int ix, int iy, bool UseNew, double Fx);
    double Jy(int ix, int iy, bool UseNew, double Fy);
    double Fi(double Ux0, double Uy0, double Fx, double Fy, int i);

    // --------------- Equilibrium functions ---------------

    double feq(double rho0, double Ux0, double Uy0, int i);
    void Start(double rho0, double UJx0, double Uy0);
    void Collision(double gx, double gy, Body *Bodies);
    void ImposeFields(void);
    void Advection(void);
    void Print(const char *Filename, double gx, double gy, double Vx0, Body *Bodies);

    // --------------- Interpolation and Spreading ---------------

    vector3D InterpolateVelocity(int n, double gx, double gy, Body &C);
    vector3D InterpolateJ(int n, double gx, double gy, Body &C);
    double InterpolateRho(int n, double gx, double gy, Body &C);
    vector3D SpreadForce(int ix, int iy, Body &C);
    friend class Engine;
};

// --------------- Functions 'Body' class ---------------

void Body::Start(double x0, double y0, double Vx0, double Vy0, double rhom0, double h0)
{
    double thetai;
    int i;
    rhom = rhom0;
    h = h0;

    // Flat plate

    Vm = h;
    for (i = 0; i < Nm; i++)
    {
        r[i].load(x0, y0 - h / 2 + i, 0.0);
        V[i].load(Vx0, Vy0, 0.0);
    }
    ds = 1.0;

    ResetForce();
    eps.resize(Nm);
}

void Body::GetEps(void)
{
    int k, l, ix, iy, ix0, iy0;
    MatrixXd A(Nm, Nm);
    A << MatrixXd::Zero(Nm, Nm);
    for (k = 0; k < Nm; k++)
    {
        ix0 = round(r[k].x()) - Ls / 2;
        iy0 = round(r[k].y()) - Ls / 2;
        for (l = 0; l < Nm; l++)
        {
            for (ix = ix0; ix < ix0 + Ls; ix++)
                for (iy = iy0; iy < iy0 + Ls; iy++)
                {
                    A(k, l) += ds * mollifier(r[k].x() - ix, r[k].y() - iy) * mollifier(r[l].x() - ix, r[l].y() - iy);
                }
        }
    }
    eps = A.colPivHouseholderQr().solve(VectorXd::Ones(Nm));
}

void Body::Print(int t)
{ // Print the center of mass coordinates
    double x0 = 0.0;
    double y0 = 0.0;
    for (int i = 0; i < Nm; i++)
    {
        x0 += r[i].x();
        y0 += r[i].y();
    }
    x0 *= 1.0 / Nm;
    y0 *= 1.0 / Nm;
}
// --------------- Functions 'Engine' class ---------------

void Engine::GetTwoBodyForces(Body &B1, Body &B2)
{
}

void Engine::GetLatticeForces(double gx, double gy, Body &B, Body *Bodies, LatticeBoltzmann &Lattice)
{
    // Calculation of Fib[m] and total F (eq. 20)
    int i, n;
    vector3D Ffs;
    Ffs.load(0.0, 0.0, 0.0);
    B.GetEps();

    // Implementation of Li2016
    vector3D j;
    double Irho;
    for (n = 0; n < Nm; n++)
    {
        j = Lattice.InterpolateJ(n, gx, gy, B);
        Irho = Lattice.InterpolateRho(n, gx, gy, B);
        B.Fib[n] = 2 * (Irho * B.V[n] - j); // dt = 1.0 (Eq. 1)
        Ffs += B.eps[n] * B.Fib[n];         // Eq. 20
    }

    Ffs *= -B.ds / ((B.rhom - 1.0) * B.Vm); // Eq. 22
    B.AddForce(Ffs);
}

void Engine::GetAllForces(double gx, double gy, Body *Bodies, LatticeBoltzmann &Lattice)
{
    int i, j;
    vector3D gvec;
    gvec.load(gx, gy, 0);

    // Reset forces of all bodies
    for (i = 0; i < N; i++)
    {
        Bodies[i].ResetForce();
    }

    // For each pair of bodies, calculate the two-body forces
    for (i = 0; i < N; i++)
        for (j = i + 1; j < N; j++)
            GetTwoBodyForces(Bodies[i], Bodies[j]);

    // Calculate lattice forces
    for (i = 0; i < N; i++)
        GetLatticeForces(gx, gy, Bodies[i], Bodies, Lattice);

    // Gravity
    for (i = 0; i < N; i++)
        Bodies[i].AddForce(gvec);
}

// --------------- Functions 'LatticeBoltzmann' class ---------------

LatticeBoltzmann::LatticeBoltzmann(void)
{
    // Set the weights
    w[0] = 4.0 / 9;
    w[1] = w[2] = w[3] = w[4] = 1.0 / 9;
    w[5] = w[6] = w[7] = w[8] = 1.0 / 36;

    // Set the velocity vectors
    Vx[0] = 0;
    Vx[1] = 1;
    Vx[2] = 0;
    Vx[3] = -1;
    Vx[4] = 0;
    Vy[0] = 0;
    Vy[1] = 0;
    Vy[2] = 1;
    Vy[3] = 0;
    Vy[4] = -1;

    Vx[5] = 1;
    Vx[6] = -1;
    Vx[7] = -1;
    Vx[8] = 1;
    Vy[5] = 1;
    Vy[6] = 1;
    Vy[7] = -1;
    Vy[8] = -1; // Create the dynamic arrays
    int ArraySize = Lx * Ly * Q;
    f = new double[ArraySize];
    fnew = new double[ArraySize];
}

LatticeBoltzmann::~LatticeBoltzmann(void)
{
    delete[] f;
    delete[] fnew;
}

double LatticeBoltzmann::rho(int ix, int iy, bool UseNew)
{
    double sum;
    int i, n0;
    for (sum = 0, i = 0; i < Q; i++)
    {
        n0 = n(ix, iy, i);
        if (UseNew)
            sum += fnew[n0];
        else
            sum += f[n0];
    }
    return sum;
}

double LatticeBoltzmann::Jx(int ix, int iy, bool UseNew, double Fx)
{
    double sum;
    int i, n0;
    for (sum = 0, i = 0; i < Q; i++)
    {
        n0 = n(ix, iy, i);
        if (UseNew)
            sum += fnew[n0] * Vx[i];
        else
            sum += f[n0] * Vx[i];
    }
    return sum + 0.5 * Fx;
}

double LatticeBoltzmann::Jy(int ix, int iy, bool UseNew, double Fy)
{
    double sum;
    int i, n0;
    for (sum = 0, i = 0; i < Q; i++)
    {
        n0 = n(ix, iy, i);
        if (UseNew)
            sum += fnew[n0] * Vy[i];
        else
            sum += f[n0] * Vy[i];
    }
    return sum + 0.5 * Fy;
}

double LatticeBoltzmann::feq(double rho0, double Ux0, double Uy0, int i)
{
    double UdotVi = Ux0 * Vx[i] + Uy0 * Vy[i], U2 = Ux0 * Ux0 + Uy0 * Uy0;
    return rho0 * w[i] * (1 + 3 * UdotVi + 4.5 * UdotVi * UdotVi - 1.5 * U2);
}

double LatticeBoltzmann::Fi(double Ux0, double Uy0, double Fx, double Fy, int i)
{
    double UdotVi = Ux0 * Vx[i] + Uy0 * Vy[i];
    double FdotVi = Fx * Vx[i] + Fy * Vy[i], UdotF = Ux0 * Fx + Uy0 * Fy;
    return ThreeUmU2tau * w[i] * (FdotVi - UdotF + 3 * UdotVi * FdotVi);
}

void LatticeBoltzmann::Start(double rho0, double Ux0, double Uy0)
{
    int ix, iy, i, n0;
    for (ix = 0; ix < Lx; ix++) // for each cell
        for (iy = 0; iy < Ly; iy++)
            for (i = 0; i < Q; i++)
            { // on each direction
                n0 = n(ix, iy, i);
                f[n0] = feq(rho0, Ux0, Uy0, i);
            }
}

void LatticeBoltzmann::Collision(double gx, double gy, Body *Bodies)
{
    int ix, iy, i, n0, n0i, n0f, c;
    double rho0, Ux0, Uy0;
    double Fx, Fy;
    vector3D fib;
    for (ix = 1; ix < Lx - 1; ix++) // for each cell
        for (iy = 1; iy < Ly - 1; iy++)
        {
            // compute the macroscopic fields on the cell
            rho0 = rho(ix, iy, false);
            Fx = gx * rho0;
            Fy = gy * rho0;
            for (c = 0; c < N; c++)
            {
                fib = SpreadForce(ix, iy, Bodies[c]);
                Fx += fib.x();
                Fy += fib.y();
            }
            Ux0 = Jx(ix, iy, false, Fx) / rho0;
            Uy0 = Jy(ix, iy, false, Fy) / rho0;
            for (i = 0; i < Q; i++)
            { // for each velocity vector
                n0 = n(ix, iy, i);
                fnew[n0] = UmUtau * f[n0] + Utau * feq(rho0, Ux0, Uy0, i) + Fi(Ux0, Uy0, Fx, Fy, i);
            }
        }
    // Bounce back
    double D = 0.9;
    for (ix = 0; ix < Lx; ix += Lx - 1)
        for (iy = 0; iy < Ly; iy++)
        {
            n0i = n(ix, iy, 1);
            n0f = n(ix, iy, 3);
            fnew[n0i] = D * f[n0f];
            fnew[n0f] = D * f[n0i];
            n0i = n(ix, iy, 2);
            n0f = n(ix, iy, 4);
            fnew[n0i] = D * f[n0f];
            fnew[n0f] = D * f[n0i];
            n0i = n(ix, iy, 5);
            n0f = n(ix, iy, 7);
            fnew[n0i] = D * f[n0f];
            fnew[n0f] = D * f[n0i];
            n0i = n(ix, iy, 6);
            n0f = n(ix, iy, 8);
            fnew[n0i] = D * f[n0f];
            fnew[n0f] = D * f[n0i];
        }
    for (iy = 0; iy < Ly; iy += Ly - 1)
        for (ix = 0; ix < Lx; ix++)
        {
            n0i = n(ix, iy, 1);
            n0f = n(ix, iy, 3);
            fnew[n0i] = D * f[n0f];
            fnew[n0f] = D * f[n0i];
            n0i = n(ix, iy, 2);
            n0f = n(ix, iy, 4);
            fnew[n0i] = D * f[n0f];
            fnew[n0f] = D * f[n0i];
            n0i = n(ix, iy, 5);
            n0f = n(ix, iy, 7);
            fnew[n0i] = D * f[n0f];
            fnew[n0f] = D * f[n0i];
            n0i = n(ix, iy, 6);
            n0f = n(ix, iy, 8);
            fnew[n0i] = D * f[n0f];
            fnew[n0f] = D * f[n0i];
        }
}

void LatticeBoltzmann::ImposeFields(void)
{
    int i, ix, iy, n0;
    double rho0;
}

void LatticeBoltzmann::Advection(void)
{
    int ix, iy, i, ixnext, iynext, n0, n0next;
    for (ix = 0; ix < Lx; ix++) // for each cell
        for (iy = 0; iy < Ly; iy++)
            for (i = 0; i < Q; i++)
            { // on each direction
                ixnext = (ix + Vx[i] + Lx) % Lx;
                iynext = (iy + Vy[i] + Ly) % Ly; // periodic boundaries
                n0 = n(ix, iy, i);
                n0next = n(ixnext, iynext, i);
                f[n0next] = fnew[n0];
            }
}

void LatticeBoltzmann::Print(const char *Filename, double gx, double gy, double Vx0, Body *Bodies)
{
    std::ofstream MyFile;
    MyFile.open(Filename, std::ios_base::app);
    double rho0, Ux0, Uy0, Fx, Fy;
    int ix, iy, c;
    vector3D fib;
    for (ix = Lx / 4; ix < 3 * Lx / 4; ix += 1)
    {
        for (iy = Ly / 4; iy < 3 * Ly / 4; iy += 1)
        {
            rho0 = rho(ix, iy, true);
            Fx = rho0 * gx;
            Fy = rho0 * gy;
            for (c = 0; c < N; c++)
            {
                fib = SpreadForce(ix, iy, Bodies[c]);
                Fx += fib.x();
                Fy += fib.y();
            }
            Ux0 = Jx(ix, iy, true, Fx) / rho0;
            Uy0 = Jy(ix, iy, true, Fy) / rho0;
            MyFile << ix << " " << iy << " " << (Ux0 / Vx0) << " " << (Uy0 / Vx0) << std::endl;
        }
    }
    MyFile.close();
}

vector3D LatticeBoltzmann::InterpolateVelocity(int n, double gx, double gy, Body &B)
{
    double rho0, Ux0, Uy0;
    int ix, iy, ix0, iy0;
    double Fx, Fy;
    vector3D fib;
    vector3D u;
    vector3D U;
    u.load(0.0, 0.0, 0.0);
    ix0 = round(B.r[n].x()) - Ls / 2;
    iy0 = round(B.r[n].y()) - Ls / 2;
    for (ix = ix0; ix < ix0 + Ls; ix++)
        for (iy = iy0; iy < iy0 + Ls; iy++)
        {
            rho0 = rho(ix, iy, false);
            Fx = rho0 * gx;
            Fy = rho0 * gy;
            Ux0 = Jx(ix, iy, false, Fx) / rho0;
            Uy0 = Jy(ix, iy, false, Fy) / rho0;
            U.load(Ux0, Uy0, 0);
            u += U * mollifier(B.r[n].x() - ix, B.r[n].y() - iy);
        }
    return u;
}

vector3D LatticeBoltzmann::InterpolateJ(int n, double gx, double gy, Body &B)
{
    double rho0, Jx0, Jy0;
    int ix, iy, ix0, iy0;
    double Fx, Fy;
    vector3D fib;
    vector3D j;
    vector3D J;
    j.load(0.0, 0.0, 0.0);
    ix0 = round(B.r[n].x()) - Ls / 2;
    iy0 = round(B.r[n].y()) - Ls / 2;
    for (ix = ix0; ix < ix0 + Ls; ix++)
        for (iy = iy0; iy < iy0 + Ls; iy++)
        {
            rho0 = rho(ix, iy, false);
            Fx = rho0 * gx;
            Fy = rho0 * gy;
            Jx0 = Jx(ix, iy, false, Fx);
            Jy0 = Jy(ix, iy, false, Fy);
            J.load(Jx0, Jy0, 0);
            j += J * mollifier(B.r[n].x() - ix, B.r[n].y() - iy);
        }
    return j;
}

double LatticeBoltzmann::InterpolateRho(int n, double gx, double gy, Body &B)
{
    double rho0, Irho0;
    int ix, iy, ix0, iy0;
    double Fx, Fy;
    vector3D fib;
    Irho0 = 0.0;
    ix0 = round(B.r[n].x()) - Ls / 2;
    iy0 = round(B.r[n].y()) - Ls / 2;
    for (ix = ix0; ix < ix0 + Ls; ix++)
        for (iy = iy0; iy < iy0 + Ls; iy++)
        {
            rho0 = rho(ix, iy, false);
            Irho0 += rho0 * mollifier(B.r[n].x() - ix, B.r[n].y() - iy);
        }
    return Irho0;
}

vector3D LatticeBoltzmann::SpreadForce(int ix, int iy, Body &B)
{
    int i;
    vector3D fib;
    fib.load(0.0, 0.0, 0.0);
    for (i = 0; i < Nm; i++)
    {
        fib += B.Fib[i] * mollifier(B.r[i].x() - ix, B.r[i].y() - iy) * B.eps[i];
    }
    return fib;
}

// --------------- Global functions ---------------

double delta(double r)
{
    double absr = fabs(r);
    if (absr < 0.5)
        return 1.0 / 3 * (1.0 + sqrt(-3.0 * r * r + 1.0));
    else if (0.5 <= absr && absr <= 1.5)
        return 1.0 / 6 * (5.0 - 3 * absr - sqrt(-3.0 * (1.0 - absr) * (1.0 - absr) + 1.0));
    else
        return 0.0;
}
double mollifier(double r1, double r2)
{
    return delta(r1) * delta(r2);
}

// --------------- Main ---------------

int main(void)
{

    LatticeBoltzmann Fluid;
    Body Bodies[N];
    Engine IB;

    double rho0 = 1.0, rhom0 = 1.5, gx = 0.0, gy = 0.0;
    double h = Nm;
    double Vx0 = 400.0 * nu / Nm;
    int t, tmax = 4.0 * h / Vx0;
    double dt = 1.0;
    int i;

    char Filename[] = "data_flat_plate.dat";
    std::ofstream MyFile(Filename);
    MyFile.close();

    // Start

    for (i = 0; i < N; i++)
        Bodies[i].Start(Lx / 3, Ly / 2, Vx0, 0, rhom0, h);
    Fluid.Start(rho0, 0, 0);

    // Run

    for (t = 1; t <= tmax; t++)
    {
        // Evolve Lattice-Boltzmann
        Fluid.Collision(gx, gy, Bodies);
        Fluid.ImposeFields();
        Fluid.Advection();

        // Evolve bodies
        for (i = 0; i < N; i++)
            Bodies[i].Move_V(dt, 0.0);
        for (i = 0; i < N; i++)
            Bodies[i].Move_r(dt, 1.0);

        // Get Forces
        IB.GetAllForces(gx, gy, Bodies, Fluid);

        // Print
        if (t == int(0.5 * h / Vx0) || t == int(2.0 * h / Vx0) || t == int(4.0 * h / Vx0))
            Fluid.Print(Filename, gx, gy, Vx0, Bodies);
        Bodies[0].Print(t);
    }

    return 0;
}
