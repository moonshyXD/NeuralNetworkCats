#include "ImageProcessor.h"

cv::Mat ImageProcessor::loadImageFromFolder(const std::string& folderPath, int index) {
    std::string imagePath = folderPath + " (" + std::to_string(index) + ")" + ".jpg"; // Формируем путь к изображению
    cv::Mat image = cv::imread(imagePath); // Загружаем изображение сформированным путем
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl; // Выводим сообщение об ошибке, если загрузка не удалась
    }
    return image; // Возвращаем загруженное изображение
}

void ImageProcessor::clearImages(std::vector<cv::Mat>& images) {
    images.clear(); // Очищаем вектор изображений
}