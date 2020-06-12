import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
import midi_manipulation
import matplotlib.pyplot as plt
import os

class neuralnet(object):

    def __init__(self):
        pass

    def decode_midi(self, songs):
        mass = []
        for song in songs:

            for track in song:
                str_tmp = ""
                for k in range(len(track)):
                    #str_tmp += str(int(track[k]))
                    if track[k] > 0:
                        mass.append(k)
        return mass

    def error(self, test, train):
        loss = 0.00
        if len(test) == 0 or len(train) == 0:
            return 0.0
        for i in range(test.shape[1]):
            err = 0
            for j in range(test.shape[0]):
                if (test[j][i] != train[j][i]):
                    err += 1
            loss += err ** 2

        loss = loss / test.shape[1] #/10000

        return loss

    def train(self, songs, midi, start_test):
        lowest_note = midi.lowerBound #индекс самой низкой ноты на фоно
        highest_note = midi.upperBound #индекс самой высокой ноты на фоно
        note_range = highest_note-lowest_note #интервал, куда мы должны попасть

        num_timesteps  = 15 #задает количество нот в единице времени. Больше всего мороки было с ним. Не меняй!
        n_visible      = 2*note_range*num_timesteps #TРазмер видимого слоя нейросети
        n_hidden       = 50 #Размер скрытого слоя

        num_epochs = 200 #количество эпох обучения
        batch_size = 100 #Количество обучающих примеров, которые мы собираемся отправить через RBM за один раз. #не увеличивай сильно
        lr         = tf.constant(0.005, tf.float32) #



        is_error_function = 0 #флаг отрисовки функции ошибки
        if start_test == 'start': # запускаем отрисовку графика вместе с тестами. Второй график менее информативен
                                  # не будем тратить на него время
            is_error_function = 1
        is_mean_square = 0 #флаг отрисовки среднеквадратичного

        ### Тензоры для нейросети (мы же используем tensorFlow):

        x  = tf.placeholder(tf.float32, [None, n_visible], name="x") #Переменная-заполнитель, содержащая наши данные
        W  = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W") #Весовая матрица, в которой хранятся веса ребер
        bh = tf.Variable(tf.zeros([1, n_hidden],  tf.float32, name="bh")) #Вектор смещения для скрытого слоя
        bv = tf.Variable(tf.zeros([1, n_visible],  tf.float32, name="bv")) #Вектор смещения для видимого слоя

        print(x)
        print(W)
        print(bh)
        print(bv)


        #### Немного вспомогательных функций

        def sample(probs):
            #возвращаем вектор вероятностей нулей и единиц
            return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

        # функция вызывается дважды:
        # 1) Когда мы определяем шаг обновления обучения
        # 2) Когда семплируем музыку на уже подготовленном RBM
        def gibbs_sample(k):

            def gibbs_step(count, k, xk):

                hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh)) #
                xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))
                return count+1, k, xk

            #шаги Гиббса для K итераций
            ct = tf.constant(0) #counter
            [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                                gibbs_step, [ct, tf.constant(k), x])
            #Необязательная штуковина. Я поленился делать оптимизатор, потому считаем, что это точка расширения для магистратуры :)
            x_sample = tf.stop_gradient(x_sample)
            return x_sample

        ### Тренируем
        x_sample = gibbs_sample(1)
        h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
        h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))


        size_bt = tf.cast(tf.shape(x)[0], tf.float32)
        W_adder  = tf.multiply(lr/size_bt, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
        bv_adder = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
        bh_adder = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))

        updt = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]


        ### Обучаем нейросеть

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            out_train = []
            out_test = []
            loss = []
            epochs = []

            for song in songs:
                out_train.append(song)

            for epoch in tqdm(range(num_epochs)):
                min_err = 99999999 # это не хардкод. Для питона это норм
                for song in songs:
                    song = np.array(song)
                    song = song[:int(np.floor(song.shape[0]/num_timesteps)*num_timesteps)]
                    song = np.reshape(song, [int(song.shape[0]/num_timesteps), int(song.shape[1]*num_timesteps)])
                    #Тренируем RBM
                    for i in range(1, len(song), batch_size):
                        tr_x = song[i:i+batch_size]
                        session = sess.run(updt, feed_dict={x: tr_x})

                        #h_tmp = h.eval(feed_dict={x: tr_x})
                        if is_error_function == 1:
                            x_sample_tmp = x_sample.eval(feed_dict={x: tr_x})
                            err = self.error(tr_x, x_sample_tmp)      # Это ошибка по отношению к эпохам
                            min_err = min(min_err, err)

                if is_error_function == 1:
                    epochs.append(epoch)
                    loss.append(min_err)


            #Запускаем Гиббса. Здесь модель уже натренирована
            sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((500, n_visible))})
            for i in tqdm(range(sample.shape[0])):
                if not any(sample[i,:]):
                    continue

                S = np.reshape(sample[i,:], (num_timesteps, 2*note_range))
                out_test.append(S)

                #print(S)
                S = midi.check_song(S, midi.mass_check, midi.midi_count)     # Тот самый алгоритм
                midi.note_state_matrix_to_midi(S, 0, "generated/generated_chord_{}".format(i))

            #out_songs = np.reshape(out_songs, (num_timesteps, 2 * note_range))#не работает, но пробуй еще
            if is_mean_square == 1:
                out_test = self.decode_midi(out_test)                         #  Это считает среднеквадратичное!
                out_train = self.decode_midi(out_train)
                if len(out_test) > len(out_train):
                    out_test = out_test[:len(out_train)]
                elif len(out_test) < len(out_train):
                    out_train = out_train[:len(out_test)]
                tmp = 0
                for i in range(len(out_test)):
                    tmp += (out_test[i] /100 - out_train[i] / 100) ** 2
                tmp = tmp / len(out_test)
                print (tmp)

            if is_error_function == 1:
                plt.plot(epochs, loss, 'ro') #зависимость ошибки от эпохи
            #plt.xlabel = "Epochs" #не прокатило. Ищи еще
            #plt.ylabel = "Error"
            if is_mean_square == 1:
                plt.plot(out_test, list(map(lambda x: 0.01 * x +76, out_test)), out_test, out_train, 'ro') #среднеквадратичное
        if is_error_function == 1 or is_mean_square == 1:
            plt.legend()
            plt.show()
