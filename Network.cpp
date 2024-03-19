#include "Network.h"

Network::Network() {
    // ������ ������������� ����� ��� ������� ����� � ��������� ����
    // ����� ��������� ��������� �������� ��� �������
    weights_input_hidden1 = { {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9} }; // ������������� ����� ����� ������� � ������ ������� �����
    weights_hidden1_hidden2 = { {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9} }; // ������������� ����� ����� ������ � ������ ������� �����
    weights_hidden2_hidden3 = { {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9} }; // ������������� ����� ����� ������ � ������� ������� �����
    weights_hidden3_output = { 0.1, 0.2, 0.3 }; // ������������� ����� ����� ������� ������� ����� � �������� �����
}

double Network::relu(double x) {
    return std::max(0.0, x); // ������� ��������� ReLU
}

double Network::forward(const cv::Mat& image) {
    // �������������� ����������� � ������ ���������
    std::vector<double> features;
    cv::Mat resizedImage;
    cv::Mat finalImage;
    cv::resize(image, resizedImage, cv::Size(150, 150)); // �������� ������ ����������� ��� ���������� � ������ ����
    cv::cvtColor(resizedImage, finalImage, cv::COLOR_BGR2GRAY); // ��������� ����������� � ������� ������

    for (int i = 0; i < finalImage.rows; ++i) {
        for (int j = 0; j < finalImage.cols; ++j) {
            features.push_back(static_cast<double>(finalImage.at<uchar>(i, j)) / 255.0); // ����������� ������� �����������
        }
    }

    // ������ ������ ����� ������ ����
    std::vector<double> currentLayerOutputs(features);
    for (int layerIndex = 0; layerIndex < 3; ++layerIndex) { // �������� �� ������� �������� ���� � ��������� ����
        std::vector<double> nextLayerOutputs;

        const std::vector<std::vector<double>>& weights = (layerIndex == 0) ? weights_input_hidden1 : // �������� ���� ��� �������� ���� � ����������� �� ������� ����
            (layerIndex == 1) ? weights_hidden1_hidden2 :
            (layerIndex == 2) ? weights_hidden2_hidden3 :
            std::vector<std::vector<double>>(1, weights_hidden3_output);

        for (int neuronIndex = 0; neuronIndex < weights.size(); ++neuronIndex) { // �������� �� ������� ������� � ������� ����
            double weightedSum = 0.0;
            for (int weightIndex = 0; weightIndex < weights[neuronIndex].size(); ++weightIndex) { // �������� �� ������� ���� �������
                weightedSum += weights[neuronIndex][weightIndex] * currentLayerOutputs[weightIndex]; // ������� ���������� �����
            }

            // ��������� ������� ��������� (� ������ ������ ReLU)
            double neuronOutput = relu(weightedSum);
            nextLayerOutputs.push_back(neuronOutput); // ���������� ����� ������� � ��������� ����
        }

        currentLayerOutputs = nextLayerOutputs; // ��������� ����� �������� ���� ��� ���������� �������
    }

    // ���������� ����� ���������� ����
    return currentLayerOutputs[0]; // � ������ ������������ �������� ��� ����� ���� ������ ����� ���� ��������
}

void Network::back(const cv::Mat& image, double target) {
    // ������ ������
    double output = forward(image); // �������� ����� ��������� ����

    // ������������ ������
    double error = target - output;

    // ������������ ��������� ��� ��������� ����
    std::vector<double> outputGradient = { -error };

    // ������������ ��������� ��� ������� ����� �������� ��������
    std::vector<double> gradients_hidden3_hidden2;
    std::vector<double> gradients_hidden2_hidden1;

    // ������������ ��������� ��� ��������� ����
    for (int i = 0; i < weights_hidden3_output.size(); ++i) { // �������� �� ���� �������� ��������� ����
        double gradient = outputGradient[0] * relu(weights_hidden3_output[i]); // ����������� ������� ��������� ReLU
        gradients_hidden3_hidden2.push_back(gradient); // ���������� ���������
    }

    // ��������� ���� ��������� ����
    for (int j = 0; j < weights_hidden3_output.size(); ++j) { // �������� �� ���� ����� ��������� ����
        weights_hidden3_output[j] += learningRate * gradients_hidden3_hidden2[j]; // ��������� ���� � �������������� ��������� � �������� ��������
    }
    // ������������ ��������� ��� ������� �������� ����
    for (int i = 0; i < weights_hidden2_hidden3.size(); ++i) {
        double sumGradient = 0.0;
        for (int j = 0; j < gradients_hidden3_hidden2.size(); ++j) {
            sumGradient += gradients_hidden3_hidden2[j] * weights_hidden2_hidden3[j][i];
        }
        double gradient = sumGradient * relu(weights_hidden2_hidden3[0][i]); // ����������� ������� ��������� ReLU
        gradients_hidden2_hidden1.push_back({ gradient });
    }
    // ��������� ���� ������� �������� ����
    weightsUpdate(weights_hidden2_hidden3, gradients_hidden2_hidden1);

    // ������������ ��������� ��� ������� �������� ����
    std::vector<double> gradients_hidden1_input;
    for (int i = 0; i < weights_hidden1_hidden2.size(); ++i) { // �������� �� ���� �������� ������� �������� ����
        double sumGradient = 0.0;
        for (int j = 0; j < gradients_hidden2_hidden1.size(); ++j) { // �������� �� ���� ���������� ������� �������� ����
            sumGradient += gradients_hidden2_hidden1[j] * weights_hidden1_hidden2[j][i]; // ������� ����� ����������
        }
        double gradient = sumGradient * relu(weights_hidden1_hidden2[0][i]); // ����������� ������� ��������� ReLU
        gradients_hidden1_input.push_back(gradient); // ���������� ���������
    }

    // ��������� ���� ������� �������� ����
    weightsUpdate(weights_hidden1_hidden2, gradients_hidden1_input);
}

void Network::train(std::vector<cv::Mat>& images, const std::vector<double>& targets, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) { // �������� �� ���� ������ ��������
        for (int i = 0; i < images.size(); ++i) { // �������� �� ���� ������������ � ������ ������

            // �������� ��������������� ������
            back(images[i], targets[i]);
        }
    }
}

void Network::weightsUpdate(std::vector<std::vector<double>>& weightsToUpdate, const std::vector<double>& gradients) {
    for (int i = 0; i < weightsToUpdate.size(); ++i) { // �������� �� ���� ������� ������� �����
        for (int j = 0; j < weightsToUpdate[i].size(); ++j) { // �������� �� ���� ��������� ������ ������� �����
            weightsToUpdate[i][j] += learningRate * gradients[i]; // ��������� ���� � �������������� ��������� � �������� ��������
        }
    }
}