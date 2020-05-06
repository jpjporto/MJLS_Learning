#include <iostream>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdbool>
#include <vector>
#include <cfloat>
#include <random>
#include <thread>

#define EIGEN_NO_DEBUG
#include <Eigen/Dense>

#include "mat.h"

using namespace Eigen;

const int d = 3, k = 1, m = 2, T = 500, N = 2500;

const double init_r = 0.5, r_decay = 0.99;
double r, r_sq_inv;

const double gamm = 0.99;

const int max_it = 100;
const int lyap_it = 100;
const int bound_it = 1000;

const double lr = 0.01/(init_r * init_r);

Matrix<double, d, d> A0((Matrix<double, d, d>() << 0.4, 0.6, -0.1, -0.4, -0.6, 0.3, 0, 0, 1).finished());
Matrix<double, d, d> A1((Matrix<double, d, d>() << 0.9, 0.5, -0.1, 0, 1, 0, -0.1, 0.5, -0.4).finished());
Matrix<double, d, k> B0((Matrix<double, d, k>() << 1, 1, 0).finished());
Matrix<double, d, k> B1((Matrix<double, d, k>() << 1, 0, 1).finished());

Matrix<double, m, m> prob((Matrix<double, m, m>() << 0.7, 0.3, 0.4, 0.6).finished());

Matrix<double, k, d> K0, K1;

Matrix<double, d, d> Q0(MatrixXd::Identity(d, d));
Matrix<double, d, d> Q1(2 * MatrixXd::Identity(d, d));
Matrix<double, k, k> R0(MatrixXd::Identity(k, k));
Matrix<double, k, k> R1(2 * MatrixXd::Identity(k, k));

Matrix<double, d, T + 1> x_t[N];
Matrix<double, k, T> u_t[N];
Matrix<double, 1, T> c_t[N];
Matrix<int, 1, T + 1> theta[N];
Matrix<double, 1, T> b_t;

Matrix<double, d, d> A[m] = { A0, A1 };
Matrix<double, d, k> B[m] = { B0, B1 };
Matrix<double, d, d> Q[m] = { Q0, Q1 };
Matrix<double, k, k> R[m] = { R0, R1 };

const int num_threads = 5;
Matrix<double, k, m*d> gradC_thread[num_threads];
Matrix<double, d, d> Sigma_K_thread[num_threads][m];
Matrix<double, 1, T> b_thread[num_threads];

Matrix<double, 1, max_it + 1> C_k, C_k_max, C_k_min, C_k_mean;
double C_kopt;

const char *file = "Ck_bounds_baseline.mat";


// Function prototypes
void run(Matrix<double, k, m*d> &K_in, Matrix<double, k, m*d> &gradC_out, Matrix<double, m*d, m*d> &Sigma_K_out);
void cost_to_go_discounted(Matrix<double, 1, T> &c);
double optCost();
int save_data();



Matrix<double, 1, max_it + 1> solveLyap(Matrix<double, k, m*d> *K_in)
{
    Matrix<double, d, d> P0, P1, EP0, EP1, phi0, phi1;
    Matrix<double, k, d> K0, K1;
    Matrix<double, k, m*d> K;
    Matrix<double, 1, max_it + 1> C_k_out;

    for (int i = 0; i <= max_it; ++i) {
        P0.setIdentity();
        P1.setIdentity();
        K0 = K_in[i].leftCols(d);
        K1 = K_in[i].rightCols(d);

        phi0 = A[0] - B[0] * K0;
        phi1 = A[1] - B[1] * K1;
        for (int j = 0; j < lyap_it; ++j) {
            EP0 = gamm * (prob(0, 0)*P0 + prob(0, 1)*P1);
            P0 = Q[0] + K0.transpose() * R[0] * K0 + phi0.transpose() * EP0 * phi0;
            P0 = 0.5*(P0 + P0.transpose());

            EP1 = gamm * (prob(1, 0)*P0 + prob(1, 1)*P1);
            P1 = Q[1] + K1.transpose() * R[1] * K1 + phi1.transpose() * EP1 * phi1;
            P1 = 0.5*(P1 + P1.transpose());
        }

        C_k_out(i) = 0.5*(P0 + P1).trace() / 12;// -c_opt;
    }

    return C_k_out;
}


int main(int arc, char *argv[])
{
    Matrix<double, k, m*d> K_npgd;
    Matrix<double, k, m*d> K_save[max_it + 1];

    Matrix<double, k, m*d> gradC;
    Matrix<double, m*d, m*d> chi_K;

    C_k_max.setConstant(-10.);
    C_k_min.setConstant(10.);
    C_k_mean.setZero();
    K_save[0].setZero(k, m*d);

    C_kopt = optCost();
    std::cout << "Opt cost: " << C_kopt << std::endl;


    for (int j = 0; j < bound_it; ++j) {

        r = init_r;
        r_sq_inv = 1 / (r * r);

        K_npgd.setZero();
        chi_K.setZero();
        auto start = std::chrono::high_resolution_clock::now();
        for (int n = 0; n < max_it; ++n)
        {
            run(K_npgd, gradC, chi_K);

            K_npgd -= lr * pow(r, 2) * gradC * chi_K.inverse();

            if (K_npgd.array().isNaN().maxCoeff() == 1) {
                // Unstable controller, stop program
                std::cout << "Unstable controller, please try a smaller step size. Iteration number: " << n << std::endl;
                return 0;
            }

            K_save[n + 1] = K_npgd;

            r *= r_decay;
            r_sq_inv = 1 / (r * r);
        }

        if (K_npgd.array().isNaN().maxCoeff() == 0) {
            C_k = solveLyap(K_save);
            C_k_max = (Matrix<double, 1, max_it + 1>() << C_k_max.array().max(C_k.array())).finished();
            C_k_min = (Matrix<double, 1, max_it + 1>() << C_k_min.array().min(C_k.array())).finished();
            C_k_mean += C_k;
            std::cout << "Final cost: " << C_k(max_it) << ", Percent Error: " << (C_k(max_it) - C_kopt) / C_kopt * 100 << "%" << std::endl;
        }
        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Elapsed time: " << duration.count() / 1000000. << std::endl;

        std::cout << "NPG iteration number: " << j << std::endl;
    }

    C_k_mean /= bound_it;
    std::cout << "Final mean cost: " << C_k_mean(max_it) << ", Percent Error: " << (C_k_mean(max_it) - C_kopt) / C_kopt * 100 << "%" << std::endl;
    save_data();
    
    return 0;
}




void sim_trajectory(Matrix<double, k, d> *K_in, const int n)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> xRand(-0.5, 0.5);
    std::uniform_real_distribution<double> unifRand(0.0, 1.0);
    std::uniform_int_distribution<int> thetaRand(0, m - 1);
    std::normal_distribution<double> uRand(0.0, r);

    //Matrix<double, 1, T> c_i;

    int init_n = n * (N / num_threads);
    int final_n = (n + 1)*(N / num_threads);

    b_thread[n].setZero();

    for (int i = init_n; i < final_n; ++i) {
        Matrix<double, d, T + 1> &x_temp = x_t[i];
        Matrix<double, k, T> &u_temp = u_t[i];
        Matrix<double, 1, T> &c_temp = c_t[i];
        Matrix<int, 1, T + 1> &theta_temp = theta[i];

        // Simulate one trajectory
        Matrix<double, d, 1> x0((Matrix<double, d, 1>() << xRand(mt), xRand(mt), xRand(mt)).finished()); //Sample initial conditions
        x_temp->col(0) = x0;

        theta_temp(0) = thetaRand(mt);

        for (int t = 0; t < T; ++t) {
            // Sample a control action
            u_temp.col(t) = -K_in[theta_temp(t)] * x_temp.col(t) + (Matrix<double, k, 1>() << uRand(mt)).finished();

            // Calculate state-action cost
            c_temp.col(t) = x_temp.col(t).transpose() * Q[theta_temp(t)] * x_temp.col(t) + u_temp.col(t).transpose() * R[theta_temp(t)] * u_temp.col(t);

            // Compute next state
            x_temp.col(t + 1) = A[theta_temp(t)] * x_temp->col(t) + B[theta_temp(t)] * u_temp->col(t);

            double rand_num = unifRand(mt);
            if (rand_num <= prob.row(theta_temp(t))(0)) {
                theta_temp(t + 1) = 0;
            }
            else {
                theta_temp(t + 1) = 1;
            }
        }

        cost_to_go_discounted(c_temp);
        b_thread[n] += c_temp;
    }
}



void compute_grad(Matrix<double, k, d> *K_in, const int n)
{
    Matrix<double, d, d> Sigma_i[m];
    Matrix<double, k, m*d> grad_log_i;
    Matrix<double, k, m*d> gradf;
    Matrix<double, k, m*d> grad_log_it;

    Matrix<double, d, T + 1> *x_temp;
    Matrix<double, k, T> *u_temp;
    Matrix<int, 1, T + 1> *theta_temp;

    int init_n = n * (N / num_threads);
    int final_n = (n + 1)*(N / num_threads);

    gradC_thread[n].setZero();
    Sigma_K_thread[n][0].setZero();
    Sigma_K_thread[n][1].setZero();
    
    for (int i = init_n; i < final_n; ++i) {
        Matrix<double, 1, T> &C_it = c_t[i];
        Matrix<double, d, T + 1> &x_temp = x_t[i];
        Matrix<double, d, T + 1> &u_temp = u_t[i];
        Matrix<double, d, T + 1> &theta_temp = theta[i];

        grad_log_i.setZero();
        Sigma_i[0].setZero();
        Sigma_i[1].setZero();

        for (int t = 0; t < T; ++t) {
            gradf.setZero();
            gradf.segment<d>(d*(*theta_temp)(t)) = -x_temp.col(t).transpose();

            grad_log_it = pow(gamm, t) * r_sq_inv*(K_in[theta_temp(t)] * x_temp.col(t) + u_temp.col(t))*gradf;

            grad_log_i += grad_log_it * (C_it(t) - b_t(t));

            Sigma_i[theta_temp(t)] += (x_temp.col(t) * x_temp.col(t).transpose()) * pow(gamm, t);
        }

        Sigma_K_thread[n][0] += Sigma_i[0];
        Sigma_K_thread[n][1] += Sigma_i[1];
        gradC_thread[n] += grad_log_i;
    }
}

void run(Matrix<double, k, m*d> &K_in, Matrix<double, k, m*d> &gradC_out, Matrix<double, m*d, m*d> &Sigma_K_out)
{
    gradC_out.setZero();
    Sigma_K_out.setZero();
    b_t.setZero();

    Matrix<double, k, d> K[m];
    // Assumes m = 2;
    K[0] = K_in.leftCols<d>();
    K[1] = K_in.rightCols<d>();
    
    std::vector<std::thread> threads;

    // Simulate trajectories and store them in memory
    for (int n = 0; n < num_threads; ++n) threads.push_back(std::thread(sim_trajectory, K, n));
    for (auto& th : threads) th.join();
    for (int n = 0; n < num_threads; ++n) b_t += b_thread[n] / N;

    threads.clear();

    // Compute gradient and state covariance estimates
    for (int n = 0; n < num_threads; ++n) threads.push_back(std::thread(compute_grad, K, n));
    for (auto& th : threads) th.join();
    for (int n = 0; n < num_threads; ++n) {
        gradC_out += gradC_thread[n] / N;
        Sigma_K_out.topLeftCorner<d, d>() += Sigma_K_thread[n][0] / N;
        Sigma_K_out.bottomRightCorner<d, d>() += Sigma_K_thread[n][1] / N;
    }
}


void cost_to_go_discounted(Matrix<double, 1, T> &c)
{
    for (int i = T - 2; i >= 0; --i) c(i) += gamm * c(i + 1);
}

double optCost()
{
    Matrix<double, d, d> P0, P1, EP0, EP1;
    Matrix<double, k, k> RBPB;

    P0.setIdentity();
    P1.setIdentity();

    for (int i = 0; i < lyap_it; ++i) {
        EP0 = gamm * (prob(0, 0)*P0 + prob(0, 1)*P1);
        RBPB = R[0] + B[0].transpose() * EP0 * B[0];
        P0 = Q[0] + A[0].transpose() * EP0 * A[0] - A[0].transpose() * EP0 * B[0] * RBPB.inverse() * B[0].transpose() * EP0 * A[0];
        P0 = 0.5*(P0 + P0.transpose());

        EP1 = gamm * (prob(1, 0)*P0 + prob(1, 1)*P1);
        RBPB = R[1] + B[1].transpose() * EP1 * B[1];
        P1 = Q[1] + A[1].transpose() * EP1 * A[1] - A[1].transpose() * EP1 * B[1] * RBPB.inverse() * B[1].transpose() * EP1 * A[1];
        P1 = 0.5*(P1 + P1.transpose());
    }

    return 0.5*(P0 + P1).trace() / 12;
}

int save_data()
{
    int status;
    MATFile *pmat;
    mxArray *pMax, *pMin, *pMean;

    pmat = matOpen(file, "w");
    if (pmat == NULL) {
        printf("Error creating file %s\n", file);
        return(EXIT_FAILURE);
    }

    pMax = mxCreateDoubleMatrix(1, max_it + 1, mxREAL);
    if (pMax == NULL) {
        printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
        printf("Unable to create mxArray.\n");
        return(EXIT_FAILURE);
    }
    memcpy((void *)(mxGetPr(pMax)), (void *)C_k_max.data(), C_k_max.size() * sizeof(double));
    status = matPutVariable(pmat, ("C_k_" + std::to_string(N) + "_max_b").c_str(), pMax);
    if (status != 0) {
        printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
        return(EXIT_FAILURE);
    }

    pMin = mxCreateDoubleMatrix(1, max_it + 1, mxREAL);
    if (pMin == NULL) {
        printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
        printf("Unable to create mxArray.\n");
        return(EXIT_FAILURE);
    }
    memcpy((void *)(mxGetPr(pMin)), (void *)C_k_min.data(), C_k_min.size() * sizeof(double));
    status = matPutVariable(pmat, ("C_k_" + std::to_string(N) + "_min_b").c_str(), pMin);
    if (status != 0) {
        printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
        return(EXIT_FAILURE);
    }

    pMean = mxCreateDoubleMatrix(1, max_it + 1, mxREAL);
    if (pMean == NULL) {
        printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
        printf("Unable to create mxArray.\n");
        return(EXIT_FAILURE);
    }
    memcpy((void *)(mxGetPr(pMean)), (void *)C_k_mean.data(), C_k_mean.size() * sizeof(double));
    status = matPutVariable(pmat, ("C_k_" + std::to_string(N) + "_mean_b").c_str(), pMean);
    if (status != 0) {
        printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
        return(EXIT_FAILURE);
    }

    // clean up
    mxDestroyArray(pMax);
    mxDestroyArray(pMin);
    mxDestroyArray(pMean);
    if (matClose(pmat) != 0) {
        printf("Error closing file %s\n", file);
        return(EXIT_FAILURE);
    }
    
    return(EXIT_SUCCESS);
}