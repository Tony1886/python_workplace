# -*- coding: utf-8 -*-


pos_train_num = 0
neg_train_num = 0
f1 = open("bezdekIris.txt", "r")
f2 = open("iris_train.txt", "w")
f3 = open("iris_test.txt", "w")
for line in f1:
    x1, x2, x3, x4, cate = line.strip().split(",")
    if cate=="Iris-setosa":
        if neg_train_num<40:
            f2.write("{} 1:{} 2:{} 3:{} 4:{}\n".format(0, x1, x2, x3, x4))
            neg_train_num += 1
        else:
            f3.write("{} 1:{} 2:{} 3:{} 4:{}\n".format(0, x1, x2, x3, x4))
    if cate=="Iris-versicolor":
        if pos_train_num<40:
            f2.write("{} 1:{} 2:{} 3:{} 4:{}\n".format(1, x1, x2, x3, x4))
            pos_train_num += 1
        else:
            f3.write("{} 1:{} 2:{} 3:{} 4:{}\n".format(1, x1, x2, x3, x4))
f1.close()
f2.close()
f3.close()