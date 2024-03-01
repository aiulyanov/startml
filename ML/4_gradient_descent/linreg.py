import numpy as np
import pandas as pd


class GradientDescentMse:
    """
    Базовый класс для реализации градиентного спуска в задаче линейной МНК регрессии
    """

    def __init__(
        self,
        samples: pd.DataFrame,
        targets: pd.DataFrame,
        learning_rate: float = 1e-3,
        threshold=1e-6,
        copy: bool = True,
    ):
        self.samples = samples
        self.targets = targets
        self.beta = np.ones(self.samples.shape[1])
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.copy = copy
        self.iteration_loss_dict = {}

    def add_constant_feature(self):
        """
        Метод для создания константной фичи в матрице объектов samples
        Метод создает колонку с константным признаком (interсept) в матрице признаков.
        Hint: так как количество признаков увеличилось на одну, не забудьте дополнить вектор с изначальными весами модели!
        """
        self.samples["bias"] = 1
        self.beta = np.append(self.beta, 1)

    def calculate_mse_loss(self) -> float:
        """
        Метод для расчета среднеквадратической ошибки

        :return: среднеквадратическая ошибка при текущих весах модели : float
        """
        return np.mean((self.targets - np.dot(self.samples, self.beta)) ** 2)

    def calculate_gradient(self) -> np.ndarray:
        """
        Метод для вычисления вектора-градиента
        Метод возвращает вектор-градиент, содержащий производные по каждому признаку.
        Сначала матрица признаков скалярно перемножается на вектор self.beta, и из каждой колонки
        полученной матрицы вычитается вектор таргетов. Затем полученная матрица скалярно умножается на матрицу признаков.
        Наконец, итоговая матрица умножается на 2 и усредняется по каждому признаку.

        :return: вектор-градиент, т.е. массив, содержащий соответствующее количество производных по каждой переменной : np.ndarray
        """

        return (
            2
            / self.samples.shape[0]
            * (np.dot(self.samples, np.dot(self.samples, self.beta) - self.targets))
        )

    def iteration(self):
        """
        Обновляем веса модели в соответствии с текущим вектором-градиентом
        """
        self.beta -= self.learning_rate * self.calculate_gradient()

    def learn(self):
        """
        Итеративное обучение весов модели до срабатывания критерия останова
        Запись mse и номера итерации в iteration_loss_dict
        """
        iter_count = 0
        start_betas = self.beta
        previous_mse = self.calculate_mse_loss()
        self.iteration()
        new_betas = self.beta
        next_mse = self.calculate_mse_loss()

        while (np.abs(new_betas - start_betas) > self.threshold) or (
            np.abs(previous_mse - next_mse) > self.threshold
        ):
            self.iteration()
            new_betas = self.beta
            next_mse = self.calculate_mse_loss()
            iter_count += 1
            if iter_count == 2:
                break


if __name__ == "__main__":
    GD = GradientDescentMse()
    print(GD.beta)