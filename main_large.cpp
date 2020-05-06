#include <iostream>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdbool>
#include <vector>
#include <cfloat>
#include <random>
#include <thread>
#include <mutex>

#define EIGEN_NO_DEBUG
#include <Eigen/Dense>

#include "mat.h"

using namespace Eigen;


const int d = 100, k = 20, m = 100, T = 500, N = 50000;
const int A_size = d * d, B_size = d * k, Q_size = d * d, R_size = k * k;

const double init_r = 1.0, r_decay = 0.99;
double r, r_sq_inv;

const double gamm = 0.99;

const int max_it = 100;
const int lyap_it = 200;
const int mini_batch = 125;
const int bound_it = 1;

const double lr = 0.00025/(init_r * init_r);


Matrix<double, Dynamic, Dynamic> A[m], B[m], Q[m], R[m];
Matrix<double, m, m> prob, trans_prob;

// Multi-thread variables
const int num_threads = 8;
Matrix<double, Dynamic, Dynamic> gradC_thread[num_threads];
Matrix<double, Dynamic, Dynamic> Sigma_K_thread[num_threads][m];

Matrix<double, 1, T> b_t;
std::mutex b_calc;
int b_count;

Matrix<double, 1, max_it + 1> C_k, C_k_max, C_k_min, C_k_mean;
double C_kopt, C_k0;

const char *sys_file = "sys_data%d_%d_%dmodes.mat"; //System matrices file
const char *cost_file = "cost_100.mat"; // cost file, not necessary
const char *save_file = "Ck_bounds_baseline_random.mat"; // save filename


// Function prototypes
Matrix<double, 1, max_it + 1> solveLyap(Matrix<double, Dynamic, Dynamic> *K_in);
void compute_grad(Matrix<double, k, Dynamic> *K_in, const int n);
void run(Matrix<double, Dynamic, Dynamic> &K_in, Matrix<double, Dynamic, Dynamic> &gradC_out, Matrix<double, Dynamic, Dynamic> *Sigma_K_out);
void runExactPG(Matrix<double, Dynamic, Dynamic> &K_in, Matrix<double, Dynamic, Dynamic> &gradC_out, Matrix<double, Dynamic, Dynamic> *Sigma_K_out);
void cost_to_go_discounted(Matrix<double, 1, T> &c);
void optCost();
void k0Cost();
int init_sys();
int save_data();
int load_cost();


int main(int arc, char *argv[])
{
    if (init_sys()) {
        // if fails reading matrix file, exit main program
        std::cout << "Failed" << std::endl;
        return 0;
    }
    std::cout << "Done" << std::endl;

    Matrix<double, Dynamic, Dynamic> K_npgd(k, m*d);
    Matrix<double, Dynamic, Dynamic> K_save[max_it + 1];

    Matrix<double, Dynamic, Dynamic> gradC(k, m*d);
    Matrix<double, Dynamic, Dynamic> SigmaK[m];

    Matrix<double, k, Dynamic> LK;

    C_k_max.setConstant(0.);
    C_k_min.setConstant(DBL_MAX);
    C_k_mean.setZero();

    for (int n = 0; n < m; n++) SigmaK[n].setZero(d, d);
    gradC.setZero(k, m*d);
    LK.setZero(k, m*d);
    K_save[0].setZero(k, m*d);

    if (load_cost()) {
        // if fails reading or cost file doesn't exists, then calculate values
        auto tic = std::chrono::high_resolution_clock::now();
        optCost();
        k0Cost();
        auto toc = std::chrono::high_resolution_clock::now();
        auto durationCost = std::chrono::duration_cast<std::chrono::microseconds>(toc - tic);
        std::cout << "Elapsed time: " << durationCost.count() / 1000000. << std::endl; //Time in seconds
    }
    std::cout << "Opt cost: " << C_kopt << ", K0 cost: " << C_k0 << ", Percent Error: " << (C_k0 - C_kopt) / C_kopt * 100 << "%" << std::endl;
    setNbThreads(1);


    for (int j = 0; j < bound_it; j++) {

        r = init_r;
        r_sq_inv = 1 / (r * r);

        K_npgd.setZero(k, m*d);
        auto start1 = std::chrono::high_resolution_clock::now();
        for (int n = 0; n < max_it; ++n) {
            run(K_npgd, gradC, SigmaK);

            for (int i = 0; i < m; i++) LK.block(0, i*d, k, d) = gradC.block(0, i*d, k, d) * SigmaK[i].inverse();

            K_npgd -= lr * pow(r, 2) * LK;

            if (K_npgd.array().isNaN().maxCoeff() == 1) {
                // Unstable controller, stop program
                std::cout << "Unstable controller, please try a smaller step size. Iteration number: " << n << std::endl;
                return 0;
            }

            K_save[n + 1] = K_npgd;

            r *= r_decay;
            r_sq_inv = 1 / (r * r);
        }
        auto stop1 = std::chrono::high_resolution_clock::now();
        auto durationNPGD = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
        std::cout << "NPGD time: " << durationNPGD.count() / 1000000. << std::endl; //Time in seconds

        auto start2 = std::chrono::high_resolution_clock::now();
        if (K_npgd.array().isNaN().maxCoeff() == 0) {
            C_k = solveLyap(K_save);
            C_k_max = (Matrix<double, 1, (max_it + 1)>() << C_k_max.array().max(C_k.array())).finished();
            C_k_min = (Matrix<double, 1, (max_it + 1)>() << C_k_min.array().min(C_k.array())).finished();
            C_k_mean += C_k;
            std::cout << "Final cost: " << C_k(max_it) << ", Percent Error: " << (C_k(max_it) - C_kopt) / C_kopt * 100 << "%" << std::endl;
        }
        
        auto stop2 = std::chrono::high_resolution_clock::now();
        auto durationLyap = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
        std::cout << "Lyap time: " << durationLyap.count() / 1000000. << std::endl; //Time in seconds

        std::cout << "NPG iteration number: " << j << std::endl;
    }

    C_k_mean /= bound_it;
    std::cout << "Final mean cost: " << C_k_mean(max_it) << ", Percent Error (from mean): " << (C_k_mean(max_it) - C_kopt) / C_kopt * 100 << "%" << std::endl;
    save_data();

    return 0;
}


void run(Matrix<double, Dynamic, Dynamic> &K_in, Matrix<double, Dynamic, Dynamic> &gradC_out, Matrix<double, Dynamic, Dynamic> *Sigma_K_out)
{
    gradC_out.setZero(k, m*d);
    for (int n = 0; n < m; n++) Sigma_K_out[n].setZero(d, d);

    b_t.setZero();
    b_count = 0;

    Matrix<double, k, Dynamic> K[m];
    for (int n = 0; n < m; n++) K[n] = K_in.block(0, n*d, k, d);

    std::vector<std::thread> threads;

    for (int n = 0; n < num_threads; ++n) threads.push_back(std::thread(compute_grad, K, n));

    for (auto& th : threads) th.join();
    
    for (int n = 0; n < num_threads; ++n) {
        gradC_out += gradC_thread[n] / N;
        for (int i = 0; i < m; ++i) Sigma_K_out[i] += Sigma_K_thread[n][i] / N;
    }
}


void compute_grad(Matrix<double, k, Dynamic> *K_in, const int n)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> xRand(-0.5, 0.5);
    std::uniform_real_distribution<double> unifRand(0.0, 1.0);
    std::uniform_int_distribution<int> thetaRand(0, m - 1);
    std::normal_distribution<double> uRand(0.0, r);

    Matrix<double, Dynamic, Dynamic> x_t[mini_batch];
    Matrix<double, k, Dynamic> u_t[mini_batch];
    Matrix<double, 1, T> c_t[mini_batch];
    Matrix<int, 1, T + 1> theta[mini_batch];

    Matrix<double, Dynamic, Dynamic> *x_temp;
    Matrix<double, k, Dynamic> *u_temp;
    Matrix<double, 1, T> *c_temp;
    Matrix<int, 1, T + 1> *theta_temp;

    Matrix<double, 1, T> b_thread;
    Matrix<double, k, Dynamic> grad_log_i;
    Matrix<double, k, Dynamic> grad_log_it(k, d);
    Matrix<double, 1, T> C_it;

    Matrix<double, k, 1> rand_action;
    Matrix<double, Dynamic, 1> x0(d, 1);

    int thread_N = (N / num_threads);
    int final_n = mini_batch;
    if (thread_N < mini_batch) final_n = thread_N;

    gradC_thread[n].setZero(k, m*d);
    for (int j = 0; j < m; ++j) Sigma_K_thread[n][j].setZero(d, d);
    for (int i = 0; i < mini_batch; ++i) {
        x_t[i].setZero(d, T + 1);
        u_t[i].setZero(k, T);
    }

    do {
        b_thread.setZero();
        
        // Simulate trajectories
        for (int i = 0; i < final_n; ++i) {
            x_temp = &x_t[i];
            u_temp = &u_t[i];
            c_temp = &c_t[i];
            theta_temp = &theta[i];

            // Simulate one trajectory
            for (int j = 0; j < d; ++j) x_temp->col(0)(j) = xRand(mt); //Sample initial conditions

            (*theta_temp)(0) = thetaRand(mt);

            for (int t = 0; t < T; ++t) {
                // Sample the control actions
                for (int j = 0; j < k; ++j) rand_action(j) = uRand(mt);
                u_temp->col(t) = -K_in[(*theta_temp)(t)] * x_temp->col(t) + rand_action;

                // Calculate state-action cost
                c_temp->col(t) = x_temp->col(t).transpose() * Q[(*theta_temp)(t)] * x_temp->col(t) + u_temp->col(t).transpose() * R[(*theta_temp)(t)] * u_temp->col(t);

                // Compute next state
                x_temp->col(t + 1) = A[(*theta_temp)(t)] * x_temp->col(t) + B[(*theta_temp)(t)] * u_temp->col(t);

                // Sample the next mode
                double rand_num = unifRand(mt);
                int next_mode = 0;
                while (trans_prob.row((*theta_temp)(t))(next_mode) <= rand_num) ++next_mode;
                (*theta_temp)(t + 1) = next_mode;
            }

            cost_to_go_discounted(c_t[i]);
            b_thread += c_t[i];
        }

        b_thread /= final_n;

        // Update global expected reward
        b_calc.lock();
        b_t = (b_count*b_t + b_thread) / (b_count + 1);
        ++b_count;
        b_calc.unlock();


        for (int i = 0; i < final_n; ++i) {
            x_temp = &x_t[i];
            u_temp = &u_t[i];
            c_temp = &c_t[i];
            theta_temp = &theta[i];
            grad_log_i.setZero(k, m*d);

            for (int t = 0; t < T; ++t) {
                grad_log_it = pow(gamm, t) * r_sq_inv*(K_in[(*theta_temp)(t)] * x_temp->col(t) + u_temp->col(t))*(-x_temp->col(t).transpose());

                grad_log_i.block(0, d*(*theta_temp)(t), k, d) += grad_log_it * ((*c_temp)(t) - b_t(t));

                Sigma_K_thread[n][(*theta_temp)(t)] += (x_temp->col(t) * x_temp->col(t).transpose()) * pow(gamm, t);
            }

            gradC_thread[n] += grad_log_i;
        }

        thread_N -= mini_batch;
        if (thread_N < mini_batch) final_n = thread_N;
    } while (thread_N > 0);
}

void cost_to_go_discounted(Matrix<double, 1, T> &c)
{
    for (int i = T - 2; i >= 0; --i) c(i) += gamm * c(i + 1);
}

void optCost()
{
    setNbThreads(4);
    Matrix<double, Dynamic, Dynamic> P[m], EP;
    Matrix<double, Dynamic, Dynamic> RBPB(k, k);
    Matrix<double, Dynamic, Dynamic> BPA(k, d);
    C_kopt = 0;

    for (int i = 0; i < m; ++i) P[i].setIdentity(d, d);

    for (int j = 0; j < lyap_it; ++j) {
        for (int n = 0; n < m; ++n) {
            EP.setZero(d, d);
            for (int l = 0; l < m; ++l) EP += prob(n, l) * P[l];
            EP *= gamm;

            RBPB = R[n] + B[n].transpose() * EP * B[n];
            BPA = B[n].transpose() * EP * A[n];
            P[n] = Q[n] + A[n].transpose() * EP * A[n] - BPA.transpose() * RBPB.inverse() * BPA;
        }
    }

    for (int n = 0; n < m; ++n) C_kopt += P[n].trace() / (m * 12);
    setNbThreads(1);
}

void k0Cost()
{
    setNbThreads(4);
    Matrix<double, Dynamic, Dynamic> P[m], EP;
    C_k0 = 0;

    for (int n = 0; n < m; ++n) P[n].setIdentity(d, d);

    for (int i = 0; i < lyap_it; ++i) {
        for (int j = 0; j < m; ++j) {
            EP.setZero(d, d);
            for (int n = 0; n < m; ++n) EP += prob(j, n) * P[n];
            EP *= gamm;

            P[j] = Q[j] + A[j].transpose() * EP * A[j];
        }
    }

    for (int n = 0; n < m; n++) C_k0 += P[n].trace() / (m * 12);
    setNbThreads(1);
}

Matrix<double, 1, max_it + 1> solveLyap(Matrix<double, Dynamic, Dynamic> *K_in)
{
    setNbThreads(4);
    Matrix<double, Dynamic, Dynamic> P[m], EP, K[m], phi[m];
    Matrix<double, 1, max_it + 1> C_k_out;

    C_k_out(0) = C_k0;

    for (int i = 1; i <= max_it; ++i) {
        // Init variables
        for (int n = 0; n < m; ++n) {
            P[n].setIdentity(d, d);
            K[n] = K_in[i].block(0, n*d, k, d);
            phi[n] = A[n] - B[n] * K[n];
        }

        for (int j = 0; j < lyap_it; ++j) {
            for (int n = 0; n < m; ++n) {
                EP.setZero(d, d);
                for (int l = 0; l < m; ++l) EP += prob(n, l) * P[l];
                EP *= gamm;

                P[n] = Q[n] + K[n].transpose() * R[n] * K[n] + phi[n].transpose() * EP * phi[n];
            }
        }

        C_k_out(i) = 0;
        for (int n = 0; n < m; n++) C_k_out(i) += P[n].trace() / (m * 12);
    }

    // Turn off Eigen multithread since we already have multithread code
    setNbThreads(1);

    return C_k_out;
}

int init_sys()
{
    MATFile *pmat;
    mxArray *pA, *pB, *pQ, *pR, *pProb;

    char filename_buf[50];
    snprintf(filename_buf, sizeof(filename_buf), sys_file, d, k, m);
    std::cout << "Reading file: " << filename_buf << " ... ";
    pmat = matOpen(filename_buf, "r");
    if (pmat == NULL) {
        printf("Error reopening file %s\n", filename_buf);
        return(EXIT_FAILURE);
    }

    pA = matGetVariable(pmat, "A");
    pB = matGetVariable(pmat, "B");
    pQ = matGetVariable(pmat, "Q");
    pR = matGetVariable(pmat, "R");
    pProb = matGetVariable(pmat, "prob");
    if ((pA == NULL) || (pB == NULL) || (pQ == NULL) || (pR == NULL) || (pProb == NULL)) {
        printf("Error reading system matrices\n");
        return(EXIT_FAILURE);
    }

    auto numDim = mxGetNumberOfDimensions(pA);
    auto *dimA = mxGetDimensions(pA);
    auto *dimB = mxGetDimensions(pB);
    auto *dimProb = mxGetDimensions(pProb);
    if ((numDim != 3) || (dimA[0] != d) || (dimB[1] != k) || (dimProb[0] != m)) {
        printf("Wrong matrix dimensions\n");
        return(EXIT_FAILURE);
    }

    for (int i = 0; i < m; ++i) {
        A[i].setZero(d, d);
        B[i].setZero(d, k);
        Q[i].setZero(d, d);
        R[i].setZero(k, k);
        memcpy((void *)A[i].data(), (void *)(mxGetPr(pA) + (i * A_size)), A_size * sizeof(double));
        memcpy((void *)B[i].data(), (void *)(mxGetPr(pB) + (i * B_size)), B_size * sizeof(double));
        memcpy((void *)Q[i].data(), (void *)(mxGetPr(pQ) + (i * Q_size)), Q_size * sizeof(double));
        memcpy((void *)R[i].data(), (void *)(mxGetPr(pR) + (i * R_size)), R_size * sizeof(double));
    }
    memcpy((void *)prob.data(), (void *)(mxGetPr(pProb)), prob.size() * sizeof(double));

    trans_prob.col(0) = prob.col(0);
    for (int i = 1; i < m; ++i) trans_prob.col(i) = trans_prob.col(i - 1) + prob.col(i);

    for (int i = 1; i < m; ++i) {
        // Make sure that all probs sum to 1, fix if within numerical error.
        if (abs(trans_prob.col(m - 1)(i) - 1.0) < 1e-14) trans_prob.col(m - 1)(i) = 1.0;
    }


    if ((trans_prob.col(m - 1).array() < VectorXd::Ones(m).array()).any()) {
        printf("Probabilities don't sum to 1\n");
        return(EXIT_FAILURE);
    }

    return(EXIT_SUCCESS);
}

int save_data()
{
    // MATLAB stuff
    int status;
    MATFile *pmat;
    mxArray *pMax, *pMin, *pMean;

    pmat = matOpen(save_file, "w");
    if (pmat == NULL) {
        std::cout << "Error creating file " << save_file << std::endl;
        return(EXIT_FAILURE);
    }

    pMax = mxCreateDoubleMatrix(1, max_it + 1, mxREAL);
    if (pMax == NULL) {
        printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
        std::cout << "Unable to create mxArray." << std::endl;
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
        std::cout << "Unable to create mxArray." << std::endl;
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
        std::cout << "Unable to create mxArray." << std::endl;
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
        std::cout << "Error closing file " << save_file << std::endl;
        return(EXIT_FAILURE);
    }

    return(EXIT_SUCCESS);
}

int load_cost()
{
    MATFile *pmat;
    mxArray *pCkopt, *pCk0;

    std::cout << "Reading cost file...";
    pmat = matOpen(cost_file, "r");
    if (pmat == NULL) {
        std::cout << "Error opening file " << cost_file << std::endl;
        return(EXIT_FAILURE);
    }

    pCkopt = matGetVariable(pmat, "C_kopt");
    pCk0 = matGetVariable(pmat, "C_k0");
    if ((pCkopt == NULL) || (pCk0 == NULL)) {
        std::cout << "Error reading cost values" << std::endl;
        return(EXIT_FAILURE);
    }

    C_kopt = *mxGetPr(pCkopt);
    C_k0 = *mxGetPr(pCk0);

    std::cout << "Done" << std::endl;

    return(EXIT_SUCCESS);
}

void runExactPG(Matrix<double, Dynamic, Dynamic> &K_in, Matrix<double, Dynamic, Dynamic> &gradC_hat, Matrix<double, Dynamic, Dynamic> *Sigma_K_hat)
{
    setNbThreads(4);
    Matrix<double, Dynamic, Dynamic> Sigma_K_hat_next[m], phi[m], Pk[m], EP;
    Matrix<double, k, Dynamic> K[m];

    gradC_hat.setZero(k, m*d);

    for (int n = 0; n < m; ++n) K[n] = K_in.block(0, n*d, k, d);

    // Initialize state transition matrices
    for (int i = 0; i < m; ++i) phi[i] = A[i] - B[i] * K[i];

    // Initialize state covariance
    for (int i = 0; i < m; ++i) {
        Sigma_K_hat[i].setIdentity(d, d);
        Sigma_K_hat[i] /= 12; // covariance of unif rand number between [-0.5, 0.5] is 1/12
    }

    // Compute the state covariance
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < m; ++i) {
            Sigma_K_hat_next[i].setZero(d, d);
            for (int l = 0; l < m; ++l) Sigma_K_hat_next[i] += prob(l, i) * phi[l] * Sigma_K_hat[l] * phi[l].transpose();
        }
        
        for (int i = 0; i < m; ++i) Sigma_K_hat[i] = Sigma_K_hat_next[i];
    }

    // Compute P^K
    for (int i = 0; i < m; ++i) Pk[i].setIdentity(d, d);


    for (int j = 0; j < lyap_it; ++j) {
        for (int i = 0; i < m; ++i) {
            EP.setZero(d, d);
            for (int l = 0; l < m; ++l) EP += prob(i, l) * Pk[l];
            EP *= gamm;

            Pk[i] = Q[i] + K[i].transpose() * R[i] * K[i] + phi[i].transpose() * EP * phi[i];
        }
    }

    for (int i = 0; i < m; ++i) {
        EP.setZero(d, d);
        for (int l = 0; l < m; ++l) EP += prob(i, l) * Pk[l];
        EP *= gamm;

        gradC_hat.block(0, i*d, k, d) = 2 * ((R[i] + B[i].transpose() * EP * B[i])*K[i] - B[i].transpose() * EP * A[i]) * Sigma_K_hat[i];
    }

    setNbThreads(1);
}