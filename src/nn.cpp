#include "func.h"
#include <random>
#include <sstream>
#include <iterator>
#include <algorithm>

// Logging function implementation
void log(const std::string& message) {
    std::cout << message << std::endl;
}

// Sigmoid function and its derivative implementations
Eigen::VectorXd sigmoid(const Eigen::VectorXd& x) {
    return 1.0 / (1.0 + (-x.array()).exp());
}

Eigen::VectorXd deriv_sigmoid(const Eigen::VectorXd& x) {
    Eigen::VectorXd fx = sigmoid(x);
    return fx.array() * (1 - fx.array());
}

// Softmax function implementation
Eigen::VectorXd softmax(const Eigen::VectorXd& z) {
    Eigen::ArrayXd exp_z = (z.array() - z.maxCoeff()).exp();
    return (exp_z / exp_z.sum()).matrix();
}

// RNN loss functions implementations
double rnn_loss(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    return -(y_true.array() * y_pred.array().log()).mean();
}

Eigen::VectorXd rnn_loss_derivative(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    return y_pred - y_true;
}

// Function to tokenize text implementation
std::vector<std::string> tokenize(const std::string& text) {
    std::istringstream stream(text);
    std::vector<std::string> tokens;
    std::string word;
    while (stream >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

// Function to read data from a file implementation
std::string read_file(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        log("Error: Unable to open file.");
        return "";
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

// Neural Network Class implementation
class NeuralNetworkRecurrent {
public:
    NeuralNetworkRecurrent(double learning_rate, int epochs, int size_input, int neuron_hidden, int size_output)
        : learn_rate(learning_rate), epoch(epochs), size_input(size_input), size_output(size_output),
        num_neuron_hidden(neuron_hidden), recurcive_hidden(Eigen::VectorXd::Zero(neuron_hidden)) {

        // Initialize weights and biases
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.01);

        w1 = Eigen::MatrixXd::NullaryExpr(num_neuron_hidden, size_input, [&]() { return d(gen); });
        w_recur = Eigen::VectorXd::NullaryExpr(num_neuron_hidden, [&]() { return d(gen); });
        w2 = Eigen::MatrixXd::NullaryExpr(size_output, num_neuron_hidden, [&]() { return d(gen); });

        b1 = Eigen::VectorXd::Zero(num_neuron_hidden);
        b2 = Eigen::VectorXd::Zero(size_output);

        // Logging
        log("--------------------------------------------------");
        log("Neural network: ");
        log("--------------------------------------------------");
        log("Input: " + std::to_string(size_input));
        log("Hidden: " + std::to_string(num_neuron_hidden));
        log("Output: " + std::to_string(size_output));
        log("--------------------------------------------------");
    }

    Eigen::VectorXd feedforward(const Eigen::VectorXd& x) {
        input_data = x;
        z1 = w1 * x;

        if (recurcive_hidden.size() > 0) {
            z1 += w_recur.cwiseProduct(recurcive_hidden);
        }
        z1 += b1;

        sigmoid_hidden = sigmoid(z1);
        z2 = w2 * sigmoid_hidden + b2;
        sigmoid_output = softmax(z2);

        recurcive_hidden = sigmoid_hidden;

        return sigmoid_output;
    }

    void backpropagation(const Eigen::VectorXd& pred, const Eigen::VectorXd& target) {
        Eigen::VectorXd delta = rnn_loss_derivative(pred, target);

        Eigen::MatrixXd grad_w2 = delta * sigmoid_hidden.transpose();
        Eigen::VectorXd grad_b2 = delta;
        w2 -= learn_rate * grad_w2;
        b2 -= learn_rate * grad_b2;

        Eigen::VectorXd grad_recur = (delta.transpose() * w2).transpose().cwiseProduct(deriv_sigmoid(z1)).cwiseProduct(recurcive_hidden);
        w_recur -= learn_rate * grad_recur;

        Eigen::VectorXd delta_input = (delta.transpose() * w2).transpose().cwiseProduct(deriv_sigmoid(z1));
        Eigen::MatrixXd grad_w1 = delta_input * input_data.transpose();
        Eigen::VectorXd grad_b1 = delta_input;

        w1 -= learn_rate * grad_w1;
        b1 -= learn_rate * grad_b1;
    }

    void train(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& Y) {
        for (int ep = 0; ep < epoch; ++ep) {
            for (size_t i = 0; i < X.size(); ++i) {
                Eigen::VectorXd pred = feedforward(X.at(i));
                backpropagation(pred, Y.at(i));
            }

            if (ep % 1 == 0) {
                double error = rnn_loss(Y.back(), feedforward(X.back()));
                log("--------------------");
                log("epoch: " + std::to_string(ep) + ", error: " + std::to_string(error));
            }
        }
    }

private:
    double learn_rate;
    int epoch;
    int size_input;
    int size_output;
    int num_neuron_hidden;
    Eigen::VectorXd recurcive_hidden;

    Eigen::MatrixXd w1;
    Eigen::VectorXd w_recur;
    Eigen::MatrixXd w2;
    Eigen::VectorXd b1;
    Eigen::VectorXd b2;

    Eigen::VectorXd input_data;
    Eigen::VectorXd z1;
    Eigen::VectorXd sigmoid_hidden;
    Eigen::VectorXd z2;
    Eigen::VectorXd sigmoid_output;
};

int main() {
    try {
        // Reading data from file (change path accordingly)
        std::string file_path = "text.txt";
        std::string text_data = read_file(file_path);

        // Ensure text data is not empty
        if (text_data.empty()) {
            log("Error: File reading failed or file is empty.");
            return -1;
        }

        // Convert to lowercase
        std::transform(text_data.begin(), text_data.end(), text_data.begin(), ::tolower);

        log("data: " + text_data.substr(0, 500));

        // Tokenize text data
        std::vector<std::string> tokens = tokenize(text_data);
        log("Number of tokens: " + std::to_string(tokens.size()));

        // Create vocabulary
        std::unordered_map<std::string, int> word_indices;
        std::vector<std::string> vocabulary;
        for (const auto& token : tokens) {
            if (word_indices.find(token) == word_indices.end()) {
                word_indices[token] = static_cast<int>(vocabulary.size());
                vocabulary.push_back(token);
            }
        }
        log("Size of vocabulary: " + std::to_string(vocabulary.size()));

        // Check if vocabulary size is too large
        if (vocabulary.size() > 100000) {
            log("Warning: Vocabulary size is large, consider using word embeddings instead of one-hot encoding.");
        }

        // Convert words to indices
        std::vector<int> indices(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            indices.at(i) = word_indices.at(tokens.at(i));
        }

        // Word embeddings
        int embed_size = 100;
        std::vector<Eigen::VectorXd> word_embeddings(vocabulary.size(), Eigen::VectorXd::Zero(embed_size));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.01);

        for (auto& emb : word_embeddings) {
            emb = Eigen::VectorXd::NullaryExpr(embed_size, [&]() { return d(gen); });
        }

        // Prepare data for training
        std::vector<Eigen::VectorXd> X(indices.size() - 1);
        std::vector<Eigen::VectorXd> Y(indices.size() - 1);
        for (size_t i = 0; i < indices.size() - 1; ++i) {
            X.at(i) = word_embeddings.at(indices.at(i));
            Y.at(i) = word_embeddings.at(indices.at(i + 1));
        }

        // Initialize and train the neural network
        NeuralNetworkRecurrent network(0.1, 3, embed_size, 20, embed_size);
        network.train(X, Y);

        // Text generation
        int size_gen = 3;
        if (!X.empty() && X.size() > 3) {
            Eigen::VectorXd start_word = X.at(3);
            Eigen::VectorXd current_input = start_word;
            std::vector<std::string> generated_sequence;
            generated_sequence.push_back(tokens.at(3));

            for (int i = 0; i < size_gen - 1; ++i) {
                Eigen::VectorXd output = network.feedforward(current_input);
                int max_index;
                output.maxCoeff(&max_index);

                // Ensure max_index is within bounds
                if (max_index < 0 || max_index >= vocabulary.size()) {
                    log("Error: max_index out of range.");
                    break;
                }

                generated_sequence.push_back(vocabulary.at(max_index));
                current_input = Eigen::VectorXd::Zero(embed_size);
                current_input(max_index) = 1.0;
            }

            std::ostringstream result_stream;
            std::copy(generated_sequence.begin(), generated_sequence.end(), std::ostream_iterator<std::string>(result_stream, " "));
            log("Generated text: " + result_stream.str());
        }
        else {
            log("Error: Insufficient data for text generation.");
        }

    }
    catch (const std::exception& ex) {
        log("Exception occurred: " + std::string(ex.what()));
        return -1;
    }

    return 0;
}
