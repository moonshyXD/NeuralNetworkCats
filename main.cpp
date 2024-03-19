#include <opencv2/opencv.hpp> 
#include <iostream>
#include <vector>
#include "Network.h"
#include "ImageProcessor.h"

void trainNetwork(const std::string& folderPath, double labelValue, Network& network) {
    ImageProcessor imageProcessor;
    std::vector<cv::Mat> images;
    std::vector<double> labels;

    // Загрузка и обучение нейросети на изображениях из указанной папки
    for (int imageIndex = 1; imageIndex <= 1000; ++imageIndex) { // Перебираем изображения с индексами от 1 до 2430
        // Загрузка изображения
        cv::Mat image = imageProcessor.loadImageFromFolder(folderPath, imageIndex); // Загружаем изображение из указанной папки с текущим индексом
        if (image.empty()) { // Проверяем, удалось ли загрузить изображение
            std::cerr << "Failed to load image " << imageIndex << " from " << folderPath << std::endl; // Выводим сообщение об ошибке, если загрузка не удалась
            break; // Прерываем цикл, если загрузка не удалась
        }

        // Добавление изображения и метки в векторы
        images.push_back(image); // Добавляем загруженное изображение в вектор изображений
        labels.push_back(labelValue); // Добавляем метку в вектор меток

        int prediction = network.forward(image);
        if (prediction != labels[0]) { // если предикшн не равен верной метке
            // Обучение нейросети на загруженных изображениях
            network.train(images, labels, 1); // Обучаем нейросеть на текущем изображении
        }
        // Очистка памяти
        imageProcessor.clearImages(images); // Очищаем вектор изображений
        labels.clear(); // Очищаем вектор меток
    }

    std::cout << std::endl << std::endl << "Reading completed" << std::endl << std::endl; // Выводим сообщение о том, что чтение файла завершено
}



int main() {
    Network network; // Создаем объект класса нейронной сети

    // Первая папка с изображениями (метка 1.0)
    std::string firstFolderPath = "Assets/train/images/Picture"; // Путь к первой папке с изображениями
    std::cout << "Training..." << std::endl;
    //trainNetwork(firstFolderPath, 1.0, network); // Обучаем нейросеть на изображениях из первой папки с меткой 1.0

    // Вторая папка с изображениями (метка 0.0)
    std::string secondFolderPath = "Assets/Dog/Picture"; // Путь ко второй папке с изображениями
    std::cout << "Training..." << std::endl;
    //trainNetwork(secondFolderPath, 0.0, network); // Обучаем нейросеть на изображениях из второй папки с меткой 0.0


    // Загружаем изображение для тестирования
    cv::Mat image = cv::imread("Assets/test/images/Picture (1).jpg"); // Загружаем изображение для тестирования
    if (image.empty()) { // Проверяем, удалось ли загрузить изображение
        std::cerr << "Failed to load image." << std::endl; // Выводим сообщение об ошибке, если загрузка не удалась
        return 1; // Завершаем программу с ошибкой
    }

    // Проходим изображение через нейронную сеть
    double prediction = network.forward(image); // Предсказываем результат с помощью нейронной сети


    // Если предсказание близко к 1, то считаем, что на изображении кошка обнаружена
    if (prediction >= 0.5) { // Проверяем, превышает ли предсказание пороговое значение
        // Находим контуры кошки
        std::vector<std::vector<cv::Point>> contours; // Создаем вектор для хранения контуров
        cv::Mat grayImage; // Создаем изображение для хранения черно-белой копии
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY); // Преобразуем цветное изображение в черно-белое
        cv::findContours(grayImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // Находим внешние контуры на черно-белом изображении
        std::cout << std::endl << std::endl << "YES" << std::endl << std::endl; // Выводим сообщение о том, что кошка обнаружена

        // Рисуем контуры кошки на цветном изображении
        cv::drawContours(image, contours, -1, cv::Scalar(0, 255, 0), 2); // Рисуем контуры на цветном изображении

        // Отображаем изображение с контурами (если кошка обнаружена)
        cv::imshow("Detected Cat", image); // Отображаем изображение с контурами кошки
    }
    else {
        std::cout << std::endl << std::endl << "NO" << std::endl << std::endl;
        // Отображаем изображение с контурами (если кошка обнаружена)
        cv::imshow("Detected Cat", image); // Отображаем изображение с контурами кошки
    }

    cv::waitKey(0); // Ждем нажатия клавиши

    return 0; // Возвращаем код завершения программы
}
