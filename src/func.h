#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>

// Logging function
void log(const std::string& message);

// Sigmoid function and its derivative
Eigen::VectorXd sigmoid(const Eigen::VectorXd& x);
Eigen::VectorXd deriv_sigmoid(const Eigen::VectorXd& x);

// Softmax function
Eigen::VectorXd softmax(const Eigen::VectorXd& z);

// RNN loss functions
double rnn_loss(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);
Eigen::VectorXd rnn_loss_derivative(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);

// Function to tokenize text
std::vector<std::string> tokenize(const std::string& text);

// Function to read data from a file
std::string read_file(const std::string& file_path);

#endif
