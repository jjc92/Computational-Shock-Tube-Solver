// To compile:
// gcc -O3 -march=native -ffast-math -fopenmp 24367.c -o 24367 -lm
// To execute:
// OMP_NUM_THREADS=4 ./24367

#include <stdio.h> // For file and console I/O operations
#include <stdlib.h> // For memory allocation and deallocation
#include <string.h> // For string manipulation functions
#include <math.h> // For mathematical functions
#include <omp.h> // For OpenMP parallel programming

// Constants
#define GAMMA 1.4
#define CFL   0.8

// Aligned malloc/free
static void *aligned_malloc(size_t alignment, size_t size) {
#ifdef aligned_alloc
    // Use aligned_alloc if available
    return aligned_alloc(alignment, ((size + alignment - 1) / alignment) * alignment);
#else
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
#endif
}

// Frees memory allocated by aligned_malloc
static void aligned_free(void *p) {
    free(p);
}

// Minmod function for slope limiting
static inline double minmod(double a, double b) {
    return (a * b <= 0.0) ? 0.0 : (fabs(a) < fabs(b) ? a : b);
}

// Returns the maximum of two doubles
static inline double max_double(double a, double b) { 
    return a > b ? a : b; 
}

// Computes pressure from density, total energy, and velocity using the ideal gas law
static inline double pressure(double rho, double E, double u) {
    return (GAMMA - 1.0) * (E - 0.5 * rho * u * u);
}

// Computes the speed of sound in a gas given density and pressure
static inline double sound_speed(double rho, double p) {
    return sqrt(GAMMA * p / rho);
}

// Boundary condition application
// Enforces boundary conditions by copying adjacenct values
static inline void apply_bcs(int Res, double *a) {
    a[0]     = a[1];
    a[Res-1] = a[Res-2];
}

// Apply boundary conditions to multpe arrays 
static void apply_bcs_all(int Res, int n, double **arrays) {
    for (int j = 0; j < n; ++j) {
        apply_bcs(Res, arrays[j]);
    }
}

// Lax–Friedrichs and Lax–Wendroff functions
typedef enum { LAX_F, LAX_W } LaxType;

// Holds temporary state for Lax–Friedrichs and Lax–Wendroff
typedef struct {
    double *drho, *dm, *dE, *dtr; // Derivative buffers
    double *flux_rho, *flux_m, *flux_E, *flux_tr; // Fluxes across cell boundaries 
} SolverContext;

// Initilise context buffers
static int init_context(SolverContext *ctx, int Res, int useTracer) {
    int err = 0;
    #define ALLOC(ptr, n) do { ctx->ptr = aligned_malloc(64, (n)*sizeof *ctx->ptr); if (!ctx->ptr) err = 1; } while(0)

    ALLOC(drho,    Res);    ALLOC(dm,     Res);    ALLOC(dE,    Res);
    if (useTracer) ALLOC(dtr, Res); else ctx->dtr = NULL;

    ALLOC(flux_rho, Res-1); ALLOC(flux_m,  Res-1); ALLOC(flux_E,  Res-1);
    if (useTracer) ALLOC(flux_tr, Res-1); else ctx->flux_tr = NULL;
    #undef ALLOC

    return err;
}

// Free the memory allocated 
static void free_context(SolverContext *ctx) {
    aligned_free(ctx->drho); aligned_free(ctx->dm); aligned_free(ctx->dE);
    if (ctx->dtr) aligned_free(ctx->dtr);
    aligned_free(ctx->flux_rho); aligned_free(ctx->flux_m); aligned_free(ctx->flux_E);
    if (ctx->flux_tr) aligned_free(ctx->flux_tr);
}

// Compute time step based on CFL condition
static double compute_dt(int Res, double dx,
    const double *rho, const double *m, const double *E) {
    double max_s = 0.0;
#pragma omp parallel for reduction(max:max_s) schedule(static)
    for (int i = 0; i < Res; ++i) {
        double u = m[i] / rho[i];
        double p = pressure(rho[i], E[i], u);
        double c = sound_speed(rho[i], p);
        max_s = max_double(max_s, fabs(u) + c); // Maximum signal speed
    }
    return CFL * dx / max_s; // Compute time step based on CFL condition
}

// Compute flux residuals using HLLC method
static void computeResiduals(SolverContext *ctx,
    int Res, double dx,
    const double *restrict rho, const double *restrict m,
    const double *restrict E, const double *restrict tracer,
    int useTracer,
    double *restrict L_rho, double *restrict L_m,
    double *restrict L_E,   double *restrict L_tr)
{
    // Slope gradients using minmod limiter
#pragma omp parallel for schedule(static)
    for (int i = 1; i < Res-1; ++i) {
        ctx->drho[i] = minmod(rho[i] - rho[i-1], rho[i+1] - rho[i]);
        ctx->dm  [i] = minmod(m[i] - m[i-1], m[i+1] - m[i]);
        ctx->dE  [i] = minmod(E[i] - E[i-1], E[i+1] - E[i]);
        if (useTracer)
            ctx->dtr[i] = minmod(tracer[i] - tracer[i-1], tracer[i+1] - tracer[i]);
    }
    // Apply zero at boundaries 
    ctx->drho[0] = ctx->drho[Res-1] = ctx->dm[0] = ctx->dm[Res-1] = ctx->dE[0] = ctx->dE[Res-1] = 0.0;
    if (useTracer) ctx->dtr[0] = ctx->dtr[Res-1] = 0.0;

    // Loop over cells to compute fluxes
#pragma omp parallel for schedule(static)
    for (int i = 0; i < Res-1; ++i) {
        // Reconstruct left/right states 
        double rhoL = rho[i] + 0.5 * ctx->drho[i];
        double mL = m[i] + 0.5 * ctx->dm[i];
        double EL = E[i] + 0.5 * ctx->dE[i];
        double trL = useTracer ? tracer[i] + 0.5 * ctx->dtr[i] : 0.0;
        double rhoR = rho[i+1] - 0.5 * ctx->drho[i+1];
        double mR = m[i+1] - 0.5 * ctx->dm[i+1];
        double ER = E[i+1] - 0.5 * ctx->dE[i+1];
        double trR = useTracer ? tracer[i+1] - 0.5 * ctx->dtr[i+1] : 0.0;

        // Primative variables and wave speed
        double uL = mL / rhoL, uR = mR / rhoR;
        double pL = pressure(rhoL, EL, uL), pR = pressure(rhoR, ER, uR);
        double cL = sound_speed(rhoL, pL), cR = sound_speed(rhoR, pR);
        double SL = fmin(uL - cL, uR - cR);
        double SR = fmax(uL + cL, uR + cR);
        // Estiamte contact wave speed
        double denom = rhoL*(SL - uL) - rhoR*(SR - uR);
        double Sstar = fabs(denom) < 1e-12 ? 0.0 : (pR - pL + rhoL*uL*(SL-uL) - rhoR*uR*(SR-uR)) / denom;

        // Compute left and right fluxes 
        double FL_r = rhoL * uL, FR_r = rhoR * uR;
        double FL_m = rhoL*uL*uL + pL, FR_m = rhoR*uR*uR + pR;
        double FL_E = (EL + pL)*uL,    FR_E = (ER + pR)*uR;
        double FL_tr = useTracer ? trL * uL : 0.0;
        double FR_tr = useTracer ? trR * uR : 0.0;

        // Compute HLLC flux based wave speeds 
        double Fr, Fm, Fe, Ftr;
        if (SL >= 0) {
            // Left-going wave 
            Fr = FL_r; Fm = FL_m; Fe = FL_E; Ftr = FL_tr;
        } else if (Sstar >= 0) {
            // Left region 
            double rhoS = rhoL*(SL-uL)/(SL-Sstar);
            double ES   = ((SL-uL)*EL - pL*uL + pL*Sstar)/(SL-Sstar);
            double trS  = useTracer ? trL*(SL-uL)/(SL-Sstar) : 0.0;
            Fr  = FL_r + SL*(rhoS - rhoL);
            Fm  = FL_m + SL*(rhoS*Sstar - mL);
            Fe  = FL_E + SL*(ES    - EL);
            Ftr = useTracer ? (FL_tr + SL*(trS - trL)) : 0.0;
        } else if (SR > 0) {
            // Right region
            double rhoS = rhoR*(SR-uR)/(SR-Sstar);
            double ES   = ((SR-uR)*ER - pR*uR + pR*Sstar)/(SR-Sstar);
            double trS  = useTracer ? trR*(SR-uR)/(SR-Sstar) : 0.0;
            Fr  = FR_r + SR*(rhoS - rhoR);
            Fm  = FR_m + SR*(rhoS*Sstar - mR);
            Fe  = FR_E + SR*(ES    - ER);
            Ftr = useTracer ? (FR_tr + SR*(trS - trR)) : 0.0;
        } else {
            // Right going wave
            Fr = FR_r; Fm = FR_m; Fe = FR_E; Ftr = FR_tr;
        }
        
        // Store fluxes
        ctx->flux_rho[i] = Fr;
        ctx->flux_m  [i] = Fm;
        ctx->flux_E  [i] = Fe;
        if (useTracer) ctx->flux_tr[i] = Ftr;
    }
    // Compute residuals from flux divergence
#pragma omp parallel for schedule(static)
    for (int i = 1; i < Res-1; ++i) {
        L_rho[i] = -(ctx->flux_rho[i] - ctx->flux_rho[i-1]) / dx;
        L_m  [i] = -(ctx->flux_m  [i] - ctx->flux_m  [i-1]) / dx;
        L_E  [i] = -(ctx->flux_E  [i] - ctx->flux_E  [i-1]) / dx;
        if (useTracer) L_tr[i] = -(ctx->flux_tr[i] - ctx->flux_tr[i-1]) / dx;
    }
    // Apply boundary conditions
    L_rho[0] = L_m[0] = L_E[0] = 0.0;
    L_rho[Res-1] = L_m[Res-1] = L_E[Res-1] = 0.0;
    if (useTracer) L_tr[0] = L_tr[Res-1] = 0.0;
}

// Function to perform Lax–Friedrichs or Lax–Wendroff flux calculation
// by computing intermediste fluxes and applying them to the state
static void flux_lax(
    SolverContext *ctx, // unused in Lax
    int Res, double dx, double dt,
    const double *rho, const double *m, const double *E, const double *tracer,
    double *rho1, double *m1, double *E1, double *tr1,
    LaxType scheme)
{
    double *rhoh = NULL, *mh = NULL, *Eh = NULL, *ph = NULL;
    // Lax-Wendroff: compute half step predictor state - improves accuracy
    if (scheme == LAX_W) {
        rhoh = malloc(Res*sizeof *rhoh);
        mh = malloc(Res*sizeof *mh);
        Eh = malloc(Res*sizeof *Eh);
        ph = malloc(Res*sizeof *ph);
        if (!rhoh||!mh||!Eh||!ph) { perror("out of memory"); exit(1); }
        // Apply update to each grid point 
#pragma omp parallel for schedule(static)
        for (int i = 0; i < Res-1; ++i) {
            double u  = m[i]/rho[i], p  = pressure(rho[i], E[i], u);
            double up = m[i+1]/rho[i+1], pp = pressure(rho[i+1], E[i+1], up);
            rhoh[i] = 0.5*(rho[i]+rho[i+1]) - dt/(2*dx)*(m[i+1] - m[i]);
            mh[i] = 0.5*(m[i]+m[i+1]) - dt/(2*dx)*(((m[i+1]*up+pp)-(m[i]*u+p)));
            Eh[i] = 0.5*(E[i]+E[i+1]) - dt/(2*dx)*(((E[i+1]+pp)*up)-((E[i]+p)*u));
            ph[i] = pressure(rhoh[i], Eh[i], mh[i]/rhoh[i]);
        }
    }

#pragma omp parallel for schedule(static)
    for (int i = 1; i < Res-1; ++i) {
        // Lax-Friedrichs
        if (scheme == LAX_F) {
            // Compute left and right states
            double uL = m[i-1]/rho[i-1], pL = pressure(rho[i-1],E[i-1],uL);
            double uR = m[i+1]/rho[i+1], pR = pressure(rho[i+1],E[i+1],uR);
            rho1[i] = 0.5*(rho[i+1]+rho[i-1]) - dt/(2*dx)*(rho[i+1]*uR - rho[i-1]*uL);
            m1[i] = 0.5*(m[i+1]+m[i-1]) - dt/(2*dx)*((m[i+1]*uR+pR)-(m[i-1]*uL+pL));
            E1[i] = 0.5*(E[i+1]+E[i-1]) - dt/(2*dx)*(((E[i+1]+pR)*uR)-((E[i-1]+pL)*uL));
            if (tr1) tr1[i] = tracer[i];
        } else {
            // Lax–Wendroff update 
            rho1[i] = rho[i] - dt/dx*(mh[i] - mh[i-1]);
            m1[i] = m[i] - dt/dx*(((mh[i]*mh[i]/rhoh[i] + ph[i]) - (mh[i-1]*mh[i-1]/rhoh[i-1] + ph[i-1])));
            E1[i] = E[i] - dt/dx*(((Eh[i]+ph[i])*(mh[i]/rhoh[i]) - ((Eh[i-1]+ph[i-1])*(mh[i-1]/rhoh[i-1]))));
            if (tr1) tr1[i] = tracer[i];
        }
    }

    // Inforce boundary conditions
    double *bc_list[] = { rho1, m1, E1, tr1 };
    int bc_n = tr1 ? 4 : 3;
    apply_bcs_all(Res, bc_n, bc_list);

    if (scheme == LAX_W) {
        free(rhoh); free(mh); free(Eh); free(ph);
    }
}

// Wrappers for specific functions 
static void flux_lax_friedrichs_ctx(SolverContext *ctx,
    int Res,double dx,double dt,
    const double*r,const double*m,const double*E,const double*tr,
    double*R1,double*M1,double*E1,double*T1)
{
    flux_lax(ctx, Res, dx, dt, r, m, E, tr, R1, M1, E1, T1, LAX_F);
}

static void flux_lax_wendroff_ctx(SolverContext *ctx,
    int Res,double dx,double dt,
    const double*r,const double*m,const double*E,const double*tr,
    double*R1,double*M1,double*E1,double*T1)
{
    flux_lax(ctx, Res, dx, dt, r, m, E, tr, R1, M1, E1, T1, LAX_W);
}

// Applies 2 stage Runge-Kutta method to HLLC fluxes
// Runge-Kutta 2nd order method provides second order accuracy in time 
typedef struct {double *rho,*m,*E,*tr; } state_t; // Struct for temp state 

static void flux_HLLC_RK2_ctx(SolverContext *ctx,
    int Res,double dx,double dt,
    const double *rho, const double *m, const double *E, const double *tracer,
    double *rho1, double *m1, double *E1, double *tr1)
{
    state_t L[2], S; // Two RK stages and intermediate state
    int useTracer = (tracer != NULL);

    // Allocate memory for both RK stages and intermediate state
    for (int k = 0; k < 2; ++k) {
        L[k].rho = malloc(Res*sizeof *L[k].rho);
        L[k].m = malloc(Res*sizeof *L[k].m);
        L[k].E = malloc(Res*sizeof *L[k].E);
        L[k].tr = useTracer ? malloc(Res*sizeof *L[k].tr) : NULL;
        if (!L[k].rho||!L[k].m||!L[k].E||(useTracer&&!L[k].tr)) {
            fprintf(stderr, "Out of memoryin RK2 stage %d\n", k+1);
            exit(1);
        }
    }
    // Intermediate state
    S.rho = malloc(Res*sizeof *S.rho);
    S.m = malloc(Res*sizeof *S.m);
    S.E = malloc(Res*sizeof *S.E);
    S.tr = useTracer ? malloc(Res*sizeof *S.tr) : NULL;
    if (!S.rho||!S.m||!S.E||(useTracer&&!S.tr)) {
        fprintf(stderr, "out of memory in RK2 intermediate\n");
        exit(1);
    }

    // RK stage 1 
    computeResiduals(ctx, Res, dx, rho, m, E, tracer, useTracer,
                     L[0].rho, L[0].m, L[0].E, L[0].tr);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < Res; ++i) {
        S.rho[i] = rho[i] + dt * L[0].rho[i];
        S.m[i] = m[i] + dt * L[0].m  [i];
        S.E[i] = E[i] + dt * L[0].E  [i];
        if (useTracer) S.tr[i] = tracer[i] + dt * L[0].tr[i];
    }
    double *bc1[] = { S.rho, S.m, S.E, S.tr };
    apply_bcs_all(Res, useTracer?4:3, bc1);

    // RK stage 2 - compute residuals and intermediate state. 
    computeResiduals(ctx, Res, dx,
                     S.rho, S.m, S.E, useTracer? S.tr: NULL, useTracer,
                     L[1].rho, L[1].m, L[1].E, L[1].tr);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < Res; ++i) {
        rho1[i] = 0.5*(rho[i] + S.rho[i] + dt * L[1].rho[i]);
        m1  [i] = 0.5*(m[i] + S.m[i] + dt * L[1].m[i]);
        E1  [i] = 0.5*(E[i] + S.E[i] + dt * L[1].E[i]);
        if (useTracer) tr1[i] = 0.5*(tracer[i] + S.tr[i] + dt * L[1].tr[i]);
    }

    // Clean up the memory allocation
    for (int k = 0; k < 2; ++k) {
        free(L[k].rho); free(L[k].m); free(L[k].E);
        if (L[k].tr) free(L[k].tr);
    }
    free(S.rho); free(S.m); free(S.E); if (S.tr) free(S.tr);
}

// Generic solver driver - initilises state, applies fluxes, and tracks features then saves results
typedef void (*FluxFuncCtx)(SolverContext*,int,double,double,
    const double*,const double*,const double*,const double*,
    double*,double*,double*,double*);

void solve_generic(
    SolverContext *ctx,
    FluxFuncCtx flux, // Pointer for chosen method 
    int Res, double left_x, double right_x, // Grid size and boundaries 
    double tmax, double Disc_loc, // MAx simulation time and discontinuty location
    double left_rho, double left_vel, double left_p, // Left state - density, velocity, pressure
    double right_rho, double right_vel, double right_p, // Right state - density, velocity, pressure
    int useTracer, // Flag for tracer usage
    const char *finalOut, // Output file for final state
    const char *featFile) // Output file for feature tracking - shock, rarefaction, contact
{
    double dx = (right_x - left_x) / (Res - 1); // Grid spacing

    // Allocate state buffers
    double *rho = malloc(Res*sizeof *rho),
           *m = malloc(Res*sizeof *m),
           *E = malloc(Res*sizeof *E),
           *rho1 = malloc(Res*sizeof *rho1),
           *m1 = malloc(Res*sizeof *m1),
           *E1 = malloc(Res*sizeof *E1),
           *tracer = useTracer? malloc(Res*sizeof *tracer) : NULL,
           *t1 = useTracer? malloc(Res*sizeof *t1) : NULL,
           *x = malloc(Res*sizeof *x);
    if (!rho||!m||!E||!rho1||!m1||!E1||(useTracer&&(!tracer||!t1))||!x) {
        fprintf(stderr, "Out of memory in solve_generic\n");
        exit(EXIT_FAILURE);
    }

    // Initialise setup for sod shock tube problem
    for (int i = 0; i < Res; ++i) {
        x[i] = left_x + i*dx;
        if (x[i] < Disc_loc) {
            rho[i] = left_rho;
            m[i] = left_rho * left_vel;
            E[i] = left_p/(GAMMA-1.0) + 0.5*left_rho*left_vel*left_vel;
            if (useTracer) tracer[i] = 1.0;
        } else {
            rho[i] = right_rho;
            m[i] = right_rho * right_vel;
            E[i] = right_p/(GAMMA-1.0) + 0.5*right_rho*right_vel*right_vel;
            if (useTracer) tracer[i] = 0.0;
        }
    }
    double *init_bc[] = { rho, m, E, tracer };
    apply_bcs_all(Res, useTracer?4:3, init_bc);

    // Feature tracking file 
    FILE *ftrack = NULL;
    if (useTracer && featFile) {
        ftrack = fopen(featFile, "w");
        fprintf(ftrack, "# t shock rare contact\n");
    }

    // Time loop
    double t = 0.0;
    while (t < tmax) {
        double dt = compute_dt(Res, dx, rho, m, E);
        if (t + dt > tmax) dt = tmax - t;

        flux(ctx, Res, dx, dt, rho, m, E, tracer, rho1, m1, E1, t1);

        double *step_bc[] = { rho1, m1, E1, t1 };
        apply_bcs_all(Res, useTracer?4:3, step_bc);

        // Swap old and new states 
        #define SWAP(a,b) do { double *tmp = a; a = b; b = tmp; } while(0)
        SWAP(rho, rho1); SWAP(m, m1); SWAP(E, E1);
        if (useTracer) SWAP(tracer, t1);
        #undef SWAP

        // If tracer used - track shock, rarefaction, and contact features
        if (ftrack) {
            double contact = x[0];
            for (int i = 0; i < Res-1; ++i) {
                if ((tracer[i]-0.5)*(tracer[i+1]-0.5) < 0) {
                    double a = (0.5-tracer[i])/(tracer[i+1]-tracer[i]);
                    contact = x[i] + a*(x[i+1]-x[i]);
                    break;
                }
            }
            double shock=0, maxj=0;
            for (int i = 0; i < Res-1; ++i) {
                double pi = pressure(rho[i],E[i],m[i]/rho[i]);
                double pj = pressure(rho[i+1],E[i+1],m[i+1]/rho[i+1]);
                double jump = fabs(pj - pi);
                if (jump > maxj) { maxj = jump; shock = 0.5*(x[i]+x[i+1]); }
            }
            double rare = x[0];
            for (int i = 1; i < Res; ++i) {
                if (rho[i] < 0.98*rho[0]) { rare = x[i]; break; }
            }
            fprintf(ftrack, "%f %f %f %f\n", t, shock, rare, contact);
        }

        t += dt;
    }
    if (ftrack) fclose(ftrack);

    // Final output saved to file 
    FILE *fo = fopen(finalOut, "w");
    for (int i = 0; i < Res; ++i) {
        double u = m[i]/rho[i];
        double p = pressure(rho[i], E[i], u);
        double e = p/((GAMMA-1.0)*rho[i]);
        fprintf(fo, "%f %f %f %f %f\n", x[i], rho[i], u, p, e);
    }
    fclose(fo);

    // Free allocated memory
    free(rho); free(m); free(E);
    free(rho1); free(m1); free(E1);
    if (useTracer) { free(tracer); free(t1); }
    free(x);
}

// Main function to run the solver and define conditions
// Runs the two exampls for Figure 1 and Figure 2 with featue tracking
int main(int argc, char **argv) {
    const char *solver = (argc > 1 ? argv[1] : "HLLC"); // Options are Lax-F, Lax-W, HLLC
    FluxFuncCtx flux;
    if (!strcmp(solver, "Lax-F")) flux = flux_lax_friedrichs_ctx;
    else if (!strcmp(solver, "Lax-W")) flux = flux_lax_wendroff_ctx;
    else flux = flux_HLLC_RK2_ctx;

    int Res = 102, useTracerMax = 1; // Number of grid points and tracer flag
    SolverContext ctx; // Context for solver
    // Initialise context for solver
    if (init_context(&ctx, Res, useTracerMax)) {
        fprintf(stderr, "Out of memory initializing context\n");
        return EXIT_FAILURE;
    }
    // First simulation (classic sod problem with tracer enabled)
    solve_generic(&ctx, // Context for solver
                  flux, // Chosen flux function
                  Res, // Number of grid points
                  0.0, 1.0, // Left boundary, Right boundary
                  0.2, // Discontinuity location
                  0.3, // Max time
                  1.0, 0.75, 1.0, // Left state - density, velocity, pressure
                  0.125, 0.0, 0.1, // Right state - density, velocity, pressure
                  1, // Use tracer
                  "Fig1.txt", // Final output file
                  "shock_features_fig1.txt"); // Feature tracking file

    // Second simulation (sod problem with tracer disabled)
    solve_generic(&ctx, // Context for solver
                  flux, // Chosen flux function
                  Res, //  Number of grid points
                  0.0, 1.0, // Left boundary, Right boundary
                  0.15, // Discontinuity location
                  0.5, // Max time
                  1.0, -2.0, 0.4, // Left state - density, velocity, pressure
                  1.0,  2.0, 0.4, // Right state - density, velocity, pressure
                  0, // No tracer being used 
                  "Fig2.txt", // Final output file
                  NULL); // If tracer changed to '1', provide file name

    // Free context and print done
    free_context(&ctx);
    printf("\n All simulations done using %s solver.\n", solver);
    printf("                ---Exit---\n\n");
    return 0;
}
