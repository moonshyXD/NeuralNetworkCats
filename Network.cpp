#include "Network.h"

Network::Network() {
    // Пример инициализации весов для скрытых слоев и выходного слоя
    // Здесь приведены случайные значения для примера
    weights_input_hidden1 = { {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9} }; // Инициализация весов между входным и первым скрытым слоем
    weights_hidden1_hidden2 = { {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9} }; // Инициализация весов между первым и вторым скрытым слоем
    weights_hidden2_hidden3 = { {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9} }; // Инициализация весов между вторым и третьим скрытым слоем
    weights_hidden3_output = { 0.1, 0.2, 0.3 }; // Инициализация весов между третьим скрытым слоем и выходным слоем
}

double Network::relu(double x) {
    return std::max(0.0, x); // Функция активации ReLU
}

double Network::forward(const cv::Mat& image) {
    // Преобразование изображения в вектор признаков
    std::vector<double> features;
    cv::Mat resizedImage;
    cv::Mat finalImage;
    cv::resize(image, resizedImage, cv::Size(150, 150)); // Изменяем размер изображения для совпадения с входом сети
    cv::cvtColor(resizedImage, finalImage, cv::COLOR_BGR2GRAY); // Переводим изображение в оттенки серого

    for (int i = 0; i < finalImage.rows; ++i) {
        for (int j = 0; j < finalImage.cols; ++j) {
            features.push_back(static_cast<double>(finalImage.at<uchar>(i, j)) / 255.0); // Нормализуем пиксели изображения
        }
    }

    // Прямой проход через каждый слой
    std::vector<double> currentLayerOutputs(features);
    for (int layerIndex = 0; layerIndex < 3; ++layerIndex) { // Проходим по каждому скрытому слою и выходному слою
        std::vector<double> nextLayerOutputs;

        const std::vector<std::vector<double>>& weights = (layerIndex == 0) ? weights_input_hidden1 : // Выбираем веса для текущего слоя в зависимости от индекса слоя
            (layerIndex == 1) ? weights_hidden1_hidden2 :
            (layerIndex == 2) ? weights_hidden2_hidden3 :
            std::vector<std::vector<double>>(1, weights_hidden3_output);

        for (int neuronIndex = 0; neuronIndex < weights.size(); ++neuronIndex) { // Проходим по каждому нейрону в текущем слое
            double weightedSum = 0.0;
            for (int weightIndex = 0; weightIndex < weights[neuronIndex].size(); ++weightIndex) { // Проходим по каждому весу нейрона
                weightedSum += weights[neuronIndex][weightIndex] * currentLayerOutputs[weightIndex]; // Считаем взвешенную сумму
            }

            // Применяем функцию активации (в данном случае ReLU)
            double neuronOutput = relu(weightedSum);
            nextLayerOutputs.push_back(neuronOutput); // Записываем выход нейрона в следующий слой
        }

        currentLayerOutputs = nextLayerOutputs; // Обновляем выход текущего слоя для следующего прохода
    }

    // Возвращаем выход последнего слоя
    return currentLayerOutputs[0]; // В случае суммирования пикселей это может быть просто сумма всех пикселей
}

void Network::back(const cv::Mat& image, double target) {
    // Прямой проход
    double output = forward(image); // Получаем выход нейронной сети

    // Рассчитываем ошибку
    double error = target - output;

    // Рассчитываем градиенты для выходного слоя
    std::vector<double> outputGradient = { -error };

    // Рассчитываем градиенты для скрытых слоев обратным проходом
    std::vector<double> gradients_hidden3_hidden2;
    std::vector<double> gradients_hidden2_hidden1;

    // Рассчитываем градиенты для выходного слоя
    for (int i = 0; i < weights_hidden3_output.size(); ++i) { // Проходим по всем нейронам выходного слоя
        double gradient = outputGradient[0] * relu(weights_hidden3_output[i]); // Производная функции активации ReLU
        gradients_hidden3_hidden2.push_back(gradient); // Записываем градиенты
    }

    // Обновляем веса выходного слоя
    for (int j = 0; j < weights_hidden3_output.size(); ++j) { // Проходим по всем весам выходного слоя
        weights_hidden3_output[j] += learningRate * gradients_hidden3_hidden2[j]; // Обновляем веса с использованием градиента и скорости обучения
    }
    // Рассчитываем градиенты для второго скрытого слоя
    for (int i = 0; i < weights_hidden2_hidden3.size(); ++i) {
        double sumGradient = 0.0;
        for (int j = 0; j < gradients_hidden3_hidden2.size(); ++j) {
            sumGradient += gradients_hidden3_hidden2[j] * weights_hidden2_hidden3[j][i];
        }
        double gradient = sumGradient * relu(weights_hidden2_hidden3[0][i]); // Производная функции активации ReLU
        gradients_hidden2_hidden1.push_back({ gradient });
    }
    // Обновляем веса второго скрытого слоя
    weightsUpdate(weights_hidden2_hidden3, gradients_hidden2_hidden1);

    // Рассчитываем градиенты для первого скрытого слоя
    std::vector<double> gradients_hidden1_input;
    for (int i = 0; i < weights_hidden1_hidden2.size(); ++i) { // Проходим по всем нейронам первого скрытого слоя
        double sumGradient = 0.0;
        for (int j = 0; j < gradients_hidden2_hidden1.size(); ++j) { // Проходим по всем градиентам второго скрытого слоя
            sumGradient += gradients_hidden2_hidden1[j] * weights_hidden1_hidden2[j][i]; // Считаем сумму градиентов
        }
        double gradient = sumGradient * relu(weights_hidden1_hidden2[0][i]); // Производная функции активации ReLU
        gradients_hidden1_input.push_back(gradient); // Записываем градиенты
    }

    // Обновляем веса первого скрытого слоя
    weightsUpdate(weights_hidden1_hidden2, gradients_hidden1_input);
}

void Network::train(std::vector<cv::Mat>& images, const std::vector<double>& targets, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) { // Проходим по всем эпохам обучения
        for (int i = 0; i < images.size(); ++i) { // Проходим по всем изображениям в наборе данных

            // Обратное распространение ошибки
            back(images[i], targets[i]);
        }
    }
}

void Network::weightsUpdate(std::vector<std::vector<double>>& weightsToUpdate, const std::vector<double>& gradients) {
    for (int i = 0; i < weightsToUpdate.size(); ++i) { // Проходим по всем строкам матрицы весов
        for (int j = 0; j < weightsToUpdate[i].size(); ++j) { // Проходим по всем элементам строки матрицы весов
            weightsToUpdate[i][j] += learningRate * gradients[i]; // Обновляем веса с использованием градиента и скорости обучения
        }
    }
}