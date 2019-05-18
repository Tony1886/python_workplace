# -*- coding: utf-8 -*-
import numpy as np
import math
x = np.mat( '2,3,3,2,1,2,3,3,3,2,1,1,2,1,3,1,2;\
            1,1,1,1,1,2,2,2,2,3,3,1,2,2,2,1,1;\
            2,3,2,3,2,2,2,2,3,1,1,2,2,3,2,2,3;\
            3,3,3,3,3,3,2,3,2,3,1,1,2,2,3,1,2;\
            1,1,1,1,1,2,2,2,2,3,3,3,1,1,2,3,2;\
            1,1,1,1,1,2,2,1,1,2,1,2,1,1,2,1,1;\
            0.697,0.774,0.634,0.668,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719;\
            0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103\
            ').T
x = np.array(x)
y = np.mat('1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0')
y = np.array(y).T
'''
x = np.mat( '1,1,2,2;\
             1,2,1,2\
             ').T
x = np.array(x)
y=np.mat('0,1,1,0')
y = np.array(y).T
'''
xrow, xcol = x.shape
yrow, ycol = y.shape
print ('x: ', x.shape, x)
print ('y: ', y.shape, y)

class BP:
    def __init__(self, n_input, n_hidden_layer, n_output, learn_rate, error, n_max_train, value):
        self.n_input = n_input
        self.n_hidden_layer = n_hidden_layer
        self.n_output = n_output
        self.learn_rate = learn_rate
        self.error = error
        self.n_max_train = n_max_train

        self.v = np.random.random((self.n_input, self.n_hidden_layer))
        self.w = np.random.random((self.n_hidden_layer, self.n_output))
        self.theta0 = np.random.random(self.n_hidden_layer)
        self.theta1 = np.random.random(self.n_output)
        self.b = []
        self.yo = []
        self.x = 0
        self.y = 0
        self.lossAll = []
        self.lossAverage = 0
        self.nRight = 0
        self.value = value

    def printParam(self):
        print ('printParam')
        print ('---------------')
        print ('     v: ', self.v)
        print ('     w: ', self.w)
        print ('theta0: ', self.theta0)
        print ('theta1: ', self.theta1)
        print ('---------------')

    def init(self, x, y):
        #print 'init'
        nx = len(x)
        ny = len(y)
        self.x = x
        self.y = y
        self.b = []
        self.yo = []
        for k in range(nx):
            tmp = []
            for h in range(self.n_hidden_layer):
                tmp.append(0)
            self.b.append(tmp)
            tmp = []
            for j in range(self.n_output):
                tmp.append(0)
            self.yo.append(tmp)

    def printResult(self):
        print ('printResult')
        self.calculateLossAll()
        print ('lossAll: ', self.lossAll)
        print ('lossAverage: ', self.lossAverage)
        self.nRight = 0
        for k in range(len(self.x)):
            print (self.y[k], '----', self.yo[k])
            self.nRight += 1
            for j in range(self.n_output):
                if(self.yo[k][j] > self.value[j][0] and self.y[k][j] != self.value[j][2]):
                    self.nRight -= 1
                    break
                if(self.yo[k][j] < self.value[j][0] and self.y[k][j] != self.value[j][1]):
                    self.nRight -= 1
                    break
        print ('right rate: %d/%d'%(self.nRight, len(self.x)))

    def printProgress(self):
        print ('yo: ', self.yo)

    def calculateLoss(self, y, yo):
        #print 'calculateLoss'
        loss = 0
        for j in range(self.n_output):
            loss += (y[j] - yo[j])**2
        return loss

    def calculateLossAll(self):
        self.lossAll = []
        for k in range(len(self.x)):
            loss = self.calculateLoss(self.y[k], self.yo[k])
            self.lossAll.append(loss)

        self.lossAverage = sum(self.lossAll) / len(self.x)

    def calculateOutput(self, x, k):
        #print 'calculateOutput'
        for h in range(self.n_hidden_layer):
            tmp = 0
            for i in range(self.n_input):
                tmp += self.v[i][h] * x[i]
            self.b[k][h] = sigmoid(tmp - self.theta0[h])

        for j in range(self.n_output):
            tmp = 0
            for h in range(self.n_hidden_layer):
                tmp += self.w[h][j] * self.b[k][h]
            self.yo[k][j] = sigmoid(tmp - self.theta1[j])
        #print 'yo of x[k]', self.yo[k]
        #print ' b of x[k]', self.b[k]

        #print ' b:', self.b
        #print 'yo:', self.yo

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))

class BPStandard(BP):
    '''
        鏍囧噯bp绠楁硶灏辨槸姣忚绠椾竴涓缁冧緥灏辨洿鏂颁竴娆″弬鏁�
    '''

    def updateParam(self, k):
        #print 'updateParam: ', k
        g = []
        #print ' y: ', self.y
        #print 'yo: ', self.yo
        #print ' b: ', self.b
        for j in range(self.n_output):
            tmp = self.yo[k][j] * (1 - self.yo[k][j]) * (self.y[k][j] - self.yo[k][j])
            g.append(tmp)
        e = []
        for h in range(self.n_hidden_layer):
            tmp = 0
            for j in range(self.n_output):
                tmp += self.b[k][h] * (1.0 - self.b[k][h]) * self.w[h][j] * g[j]
            e.append(tmp)
        #print ' g: ', g
        #print ' e: ', e

        for h in range(self.n_hidden_layer):
            for j in range(self.n_output):
                self.w[h][j] += self.learn_rate * g[j] * self.b[k][h]
        for j in range(self.n_output):
            self.theta1[j] -= self.learn_rate * g[j]
        for i in range(self.n_input):
            for h in range(self.n_hidden_layer):
                self.v[i][h] += self.learn_rate * e[h] * self.x[k][i]
        for h in range(self.n_hidden_layer):
            self.theta0[h] -= self.learn_rate * e[h]


    def train(self, x, y):
        print ('train neural networks')
        self.init(x, y)
        self.printParam()
        tag = 0
        loss1 = 0
        print ('train begin:')
        n_train = 0
        nr = 0
        while 1:
            for k in range(len(x)):
                n_train += 1
                self.calculateOutput(x[k], k)
                #loss = self.calculateLoss(y[k], self.yo[k])
                self.calculateLossAll()
                loss = self.lossAverage
                #print 'k, y, yo, loss', k, y[k], self.yo[k], loss
                if abs(loss1 - loss) < self.error:
                    nr += 1
                    if nr >= 100: # 杩炵画100娆¤揪鍒扮洰鏍囨墠缁撴潫
                        break
                else:
                    nr = 0
                    self.updateParam(k)

                if n_train % 10000 == 0:
                    for k in range(len(x)):
                        self.calculateOutput(x[k], k)
                    self.printProgress()

            if n_train > self.n_max_train or nr >= 100:
                break

        print ('train end')
        self.printParam()
        self.printResult()
        print ('train count: ', n_train)

class BPAll(BP):
    def updateParam(self):
        #print 'updateParam: ', k
        g = []
        #print ' y: ', self.y
        #print 'yo: ', self.yo
        #print ' b: ', self.b
        for k in range(len(self.x)):
            gk = []
            for j in range(self.n_output):
                tmp = self.yo[k][j] * (1 - self.yo[k][j]) * (self.y[k][j] - self.yo[k][j])
                gk.append(tmp)
            g.append(gk)

        e = []
        for k in range(len(self.x)):
            ek = []
            for h in range(self.n_hidden_layer):
                tmp = 0
                for j in range(self.n_output):
                    tmp += self.b[k][h] * (1.0 - self.b[k][h]) * self.w[h][j] * g[k][j]
                ek.append(tmp)
            e.append(ek)

        #print ' g: ', g
        #print ' e: ', e

        for h in range(self.n_hidden_layer):
            for j in range(self.n_output):
                for k in range(len(self.x)):
                    self.w[h][j] += self.learn_rate * g[k][j] * self.b[k][h]
        for j in range(self.n_output):
            for k in range(len(self.x)):
                self.theta1[j] -= self.learn_rate * g[k][j]

        for i in range(self.n_input):
            for h in range(self.n_hidden_layer):
                for k in range(len(self.x)):
                    self.v[i][h] += self.learn_rate * e[k][h] * self.x[k][i]
        for h in range(self.n_hidden_layer):
            for k in range(len(self.x)):
                self.theta0[h] -= self.learn_rate * e[k][h]



    def train(self, x, y):
        print ('train neural networks')
        self.init(x, y)
        tag = 0
        loss1 = 0
        print ('train begin:')
        n_train = 0
        self.printParam()
        nr = 0
        while 1:
            n_train += 1

            for k in range(len(x)):
               self.calculateOutput(x[k], k)

            self.calculateLossAll()
            loss = self.lossAverage
            if abs(loss - loss1) < self.error:
                nr += 1
                # 杩炵画100娆¤揪鍒扮洰鏍囨墠缁撴潫
                if(nr >= 100):
                    break;
            else:
                nr = 0
                self.updateParam()
            if n_train % 10000 == 0:
                self.printProgress()
        print ('train end')
        self.printParam()
        self.printResult()
        print ('train count: ', n_train)

if __name__ == '__main__':
    # 鍙傛暟鍒嗗埆鏄� 灞炴�ф暟閲忥紝闅愬眰绁炵粡鍏冩暟閲忥紝杈撳嚭鍊兼暟閲忥紝瀛︿範鐜囷紝璇樊
    # 鏈�澶ц凯浠ｆ鏁� 浠ュ強 瀵瑰簲姣忎釜杈撳嚭鐨勫彇鍊�(鐢ㄤ簬璁＄畻姝ｇ‘鐜�)
    n_input = xcol
    n_hidden_layer = 10
    n_output = ycol
    learn_rate = 0.1
    error = 0.005
    n_max_train = 1000000
    value = [[0.5, 0, 1]]
    print('BPStandard:')
    bps = BPStandard(n_input, n_hidden_layer, n_output, learn_rate, error, n_max_train, value) 
    bps.train(x, y)
    print('BPAll:')
    bpa = BPAll(n_input, n_hidden_layer, n_output, learn_rate, error, n_max_train, value)
    bpa.train(x, y)
   