// dependencies
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <boost/numeric/odeint.hpp>
#include <vector>
#include <cmath>
namespace py = pybind11;
using namespace boost::numeric::odeint;

int num_states = 9;

// SIR model
struct SIR {
    double T_report;
    double rho_report;
    std::vector<double> beta, gamma;
    std::vector<double> beta_modifiers;

    SIR(const std::vector<double>& beta, const std::vector<double>& gamma,
        double rho_report, double T_report,
        const std::vector<double>& beta_modifiers)
        : beta(beta), gamma(gamma), rho_report(rho_report), T_report(T_report), beta_modifiers(beta_modifiers) {}

    void operator()(const std::vector<double>& y, std::vector<double>& dydt, double t) {
        int num_strains = beta.size();
        int state_size = num_states * num_strains;  
        std::vector<double> T(num_strains, 0.0);

        // Compute total population for each strain
        for (int strain = 0; strain < num_strains; ++strain) {
            int idx = strain * num_states;
            T[strain] = y[idx] + y[idx + 1] + y[idx + 2];
        }

        // Get right modifier values
        int t_int = static_cast<int>(t);
        int index = t_int + 30;
        double modifier = (index >= 0 && index < beta_modifiers.size()) ? beta_modifiers[index] : 0.0;

        // Compute SIR model for every strain
        for (int strain = 0; strain < num_strains; ++strain) {

            int idx = strain * num_states;
            double S = y[idx], I = y[idx + 1], R = y[idx + 2], I_inc_LCT0 = y[idx+3],  I_inc_LCT1 = y[idx+4], I_inc_LCT2 = y[idx+5], I_inc_LCT3 = y[idx+6], I_inc_LCT4 = y[idx+7], I_inc = y[idx+8];
            
            double beta_t = beta[strain] * (modifier+1);                                        // beta(t)
            double lambda = beta_t * S * I / T[strain];                                         // FOI

            dydt[idx] = -lambda;                                                                // dS/dt
            dydt[idx + 1] = lambda - gamma[0] * I;                                              // dI/dt
            dydt[idx + 2] = gamma[0] * I;                                                       // dR/dt
            dydt[idx + 3] = rho_report * lambda - (5/T_report) * I_inc_LCT0;                    // dI_inc_LCT0/dt
            dydt[idx + 4] = (5/T_report) * I_inc_LCT0 - (5/T_report) * I_inc_LCT1;              // dI_inc_LCT1/dt
            dydt[idx + 5] = (5/T_report) * I_inc_LCT1 - (5/T_report) * I_inc_LCT2;              // dI_inc_LCT2/dt
            dydt[idx + 6] = (5/T_report) * I_inc_LCT2 - (5/T_report) * I_inc_LCT3;              // dI_inc_LCT3/dt
            dydt[idx + 7] = (5/T_report) * I_inc_LCT3 - (5/T_report) * I_inc_LCT4;              // dI_inc_LCT4/dt
            dydt[idx + 8] = (5/T_report) * I_inc_LCT4 - I_inc;                                  // dI_inc/dt
        }
    }
};

// Gaussian smoothing function
std::vector<double> gaussian_smooth(const std::vector<double>& input, double sigma) {
    int size = static_cast<int>(sigma * 3);  // Kernel size (3 * sigma rule)
    std::vector<double> kernel(size * 2 + 1);
    std::vector<double> output(input.size(), 0.0);
    double sum = 0.0;

    // Compute Gaussian kernel
    for (int i = -size; i <= size; ++i) {
        kernel[i + size] = std::exp(-0.5 * (i * i) / (sigma * sigma));
        sum += kernel[i + size];
    }
    for (double& k : kernel) k /= sum;  // Normalize kernel

    // Convolve input with Gaussian kernel
    for (int i = 0; i < input.size(); ++i) {
        double smoothed = 0.0;
        for (int j = -size; j <= size; ++j) {
            int index = i + j;
            if (index >= 0 && index < input.size()) {
                smoothed += input[index] * kernel[j + size];
            }
        }
        output[i] = smoothed;
    }

    return output;
}

// Function to compute daily beta modifiers with padding and smoothing
std::vector<double> process_beta_modifiers(const std::vector<double>& delta_beta_temporal, 
                                           int modifier_length, int total_days, double sigma) {
    std::vector<double> daily_beta(total_days, 0.0);
    int num_modifiers = delta_beta_temporal.size();

    // Step 1: Expand modifiers to daily values
    for (int i = 0; i < num_modifiers; ++i) {
        double modifier = delta_beta_temporal[i];
        int start_day = i * modifier_length;
        for (int j = 0; j < modifier_length; ++j) {
            int day = start_day + j;
            if (day < total_days) {
                daily_beta[day] = modifier;
            }
        }
    }

    // Step 2: Pad with 30 extra days of `0.0`
    std::vector<double> padded_beta(30, 0.0);
    padded_beta.insert(padded_beta.end(), daily_beta.begin(), daily_beta.end());
    padded_beta.insert(padded_beta.end(), 30, 0.0);

    // Step 3: Apply Gaussian smoothing
    return gaussian_smooth(padded_beta, sigma);
}

// Interpolation function for output
std::vector<std::vector<double>> interpolate_results(const std::vector<std::vector<double>>& results, double t_start, double t_end) {
    std::vector<std::vector<double>> interpolated;
    int num_cols = results[0].size();
    size_t j = 0;
    for (double t = t_start; t <= t_end; ++t) {
        while (j + 1 < results.size() && results[j + 1][0] <= t) {
            ++j;
        }
        std::vector<double> interp_row(num_cols, 0.0);
        interp_row[0] = t;
        if (results[j][0] == t) {
            for (int col = 1; col < num_cols; ++col) {
                interp_row[col] = results[j][col];
            }
        } else if (j + 1 < results.size()) {
            double t1 = results[j][0], t2 = results[j + 1][0];
            for (int col = 1; col < num_cols; ++col) {
                double y1 = results[j][col], y2 = results[j + 1][col];
                interp_row[col] = y1 + (y2 - y1) * (t - t1) / (t2 - t1);
            }
        }
        interpolated.push_back(interp_row);
    }
    return interpolated;
}

// Function to integrate the SIR model
std::vector<std::vector<double>> solve(double t_start, double t_end,
                                        double atol, double rtol,
                                        std::vector<double> S0, std::vector<double> I0, std::vector<double> R0,
                                        std::vector<double> beta, std::vector<double> gamma, double rho_report, double T_report,
                                        const std::vector<double>& delta_beta_temporal, 
                                        int modifier_length, double sigma
                                        ) {
    double dt = 1;                  // initial guess for step size
    int num_strains = S0.size();    // determine number of strains
    int total_days = static_cast<int>(t_end) + 1;
    std::vector<double> beta_modifiers = process_beta_modifiers(delta_beta_temporal, modifier_length, total_days, sigma);

    // Flatten initial conditions for integration
    std::vector<double> y;
    for (int strain = 0; strain < num_strains; ++strain) {
        y.push_back(S0[strain]);
        y.push_back(I0[strain]);
        y.push_back(R0[strain]);
        y.push_back(0.0);   // I_inc_LCT0
        y.push_back(0.0);   // I_inc_LCT1
        y.push_back(0.0);   // I_inc_LCT2
        y.push_back(0.0);   // I_inc_LCT3
        y.push_back(0.0);   // I_inc_LCT4
        y.push_back(0.0);   // I_inc
    }

    // Observe only S, I, R, and I_inc: omit linear chain trick
    std::vector<std::vector<double>> results;
    auto observer = [&](const std::vector<double>& y, double t) {
        std::vector<double> row = {t};
        // perform ommission
        for (int strain = 0; strain < num_strains; ++strain) {
            int idx = strain * num_states;  
            row.push_back(y[idx + 0]);   // S
            row.push_back(y[idx + 1]);   // I
            row.push_back(y[idx + 2]);   // R
            row.push_back(y[idx + 8]);   // I_inc (skip I_inc_LCT states)
        }
        results.push_back(row);
    };
    
    SIR sir_system(beta, gamma, rho_report, T_report,  beta_modifiers);
    runge_kutta_dopri5<std::vector<double>> stepper;
    integrate_adaptive(make_controlled(atol, rtol, stepper), sir_system, y, t_start, t_end, dt, observer);
    return interpolate_results(results, t_start, t_end);
}

// Bind C++ module to Python
PYBIND11_MODULE(sir_model, m) {
    m.def("integrate", &solve, "Solve a strain-stratified SIR model",
          py::arg("t_start"), py::arg("t_end"),                                                         // solve between t_start and t_end
          py::arg("atol"), py::arg("rtol"),                                                             // solver accuracy        
          py::arg("S0"), py::arg("I0"), py::arg("R0"),                                                  // initial condition
          py::arg("beta"), py::arg("gamma"), py::arg("rho_report"), py::arg("T_report"),                // SIR parameters
          py::arg("delta_beta_temporal"), py::arg("modifier_length"), py::arg("sigma")                  // modifier parameters
          );
}
