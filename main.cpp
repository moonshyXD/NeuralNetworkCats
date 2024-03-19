#include <opencv2/opencv.hpp> 
#include <iostream>
#include <vector>
#include "Network.h"
#include "ImageProcessor.h"

void trainNetwork(const std::string& folderPath, double labelValue, Network& network) {
    ImageProcessor imageProcessor;
    std::vector<cv::Mat> images;
    std::vector<double> labels;

    // �������� � �������� ��������� �� ������������ �� ��������� �����
    for (int imageIndex = 1; imageIndex <= 1000; ++imageIndex) { // ���������� ����������� � ��������� �� 1 �� 2430
        // �������� �����������
        cv::Mat image = imageProcessor.loadImageFromFolder(folderPath, imageIndex); // ��������� ����������� �� ��������� ����� � ������� ��������
        if (image.empty()) { // ���������, ������� �� ��������� �����������
            std::cerr << "Failed to load image " << imageIndex << " from " << folderPath << std::endl; // ������� ��������� �� ������, ���� �������� �� �������
            break; // ��������� ����, ���� �������� �� �������
        }

        // ���������� ����������� � ����� � �������
        images.push_back(image); // ��������� ����������� ����������� � ������ �����������
        labels.push_back(labelValue); // ��������� ����� � ������ �����

        int prediction = network.forward(image);
        if (prediction != labels[0]) { // ���� �������� �� ����� ������ �����
            // �������� ��������� �� ����������� ������������
            network.train(images, labels, 1); // ������� ��������� �� ������� �����������
        }
        // ������� ������
        imageProcessor.clearImages(images); // ������� ������ �����������
        labels.clear(); // ������� ������ �����
    }

    std::cout << std::endl << std::endl << "Reading completed" << std::endl << std::endl; // ������� ��������� � ���, ��� ������ ����� ���������
}



int main() {
    Network network; // ������� ������ ������ ��������� ����

    // ������ ����� � ������������� (����� 1.0)
    std::string firstFolderPath = "Assets/train/images/Picture"; // ���� � ������ ����� � �������������
    std::cout << "Training..." << std::endl;
    //trainNetwork(firstFolderPath, 1.0, network); // ������� ��������� �� ������������ �� ������ ����� � ������ 1.0

    // ������ ����� � ������������� (����� 0.0)
    std::string secondFolderPath = "Assets/Dog/Picture"; // ���� �� ������ ����� � �������������
    std::cout << "Training..." << std::endl;
    //trainNetwork(secondFolderPath, 0.0, network); // ������� ��������� �� ������������ �� ������ ����� � ������ 0.0


    // ��������� ����������� ��� ������������
    cv::Mat image = cv::imread("Assets/test/images/Picture (1).jpg"); // ��������� ����������� ��� ������������
    if (image.empty()) { // ���������, ������� �� ��������� �����������
        std::cerr << "Failed to load image." << std::endl; // ������� ��������� �� ������, ���� �������� �� �������
        return 1; // ��������� ��������� � �������
    }

    // �������� ����������� ����� ��������� ����
    double prediction = network.forward(image); // ������������� ��������� � ������� ��������� ����


    // ���� ������������ ������ � 1, �� �������, ��� �� ����������� ����� ����������
    if (prediction >= 0.5) { // ���������, ��������� �� ������������ ��������� ��������
        // ������� ������� �����
        std::vector<std::vector<cv::Point>> contours; // ������� ������ ��� �������� ��������
        cv::Mat grayImage; // ������� ����������� ��� �������� �����-����� �����
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY); // ����������� ������� ����������� � �����-�����
        cv::findContours(grayImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // ������� ������� ������� �� �����-����� �����������
        std::cout << std::endl << std::endl << "YES" << std::endl << std::endl; // ������� ��������� � ���, ��� ����� ����������

        // ������ ������� ����� �� ������� �����������
        cv::drawContours(image, contours, -1, cv::Scalar(0, 255, 0), 2); // ������ ������� �� ������� �����������

        // ���������� ����������� � ��������� (���� ����� ����������)
        cv::imshow("Detected Cat", image); // ���������� ����������� � ��������� �����
    }
    else {
        std::cout << std::endl << std::endl << "NO" << std::endl << std::endl;
        // ���������� ����������� � ��������� (���� ����� ����������)
        cv::imshow("Detected Cat", image); // ���������� ����������� � ��������� �����
    }

    cv::waitKey(0); // ���� ������� �������

    return 0; // ���������� ��� ���������� ���������
}
