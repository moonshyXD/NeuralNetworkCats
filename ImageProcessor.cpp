#include "ImageProcessor.h"

cv::Mat ImageProcessor::loadImageFromFolder(const std::string& folderPath, int index) {
    std::string imagePath = folderPath + " (" + std::to_string(index) + ")" + ".jpg"; // ��������� ���� � �����������
    cv::Mat image = cv::imread(imagePath); // ��������� ����������� �������������� �����
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl; // ������� ��������� �� ������, ���� �������� �� �������
    }
    return image; // ���������� ����������� �����������
}

void ImageProcessor::clearImages(std::vector<cv::Mat>& images) {
    images.clear(); // ������� ������ �����������
}