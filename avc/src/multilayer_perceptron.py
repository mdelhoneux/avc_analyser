import random
import numpy as np

class MLP(object):
    def __init__(self, in_dim, hid_dim, out_dim, labels,epochs=100):
        #TODO: check how necessary this is
        import dynet as dy
        global dy
        #dy.reset_random_seed(int(''.join([str(random.randint(0,9)) for i in range(5)])))
        self.model = dy.ParameterCollection()
        dy.renew_cg()
        #self.trainer = dy.AdamTrainer(self.model,learning_rate=0.01)
        self.trainer = dy.AdamTrainer(self.model)
        self._W1 = self.model.add_parameters((hid_dim, in_dim))
        self._b1 = self.model.add_parameters(hid_dim)
        self._W2 = self.model.add_parameters((out_dim, hid_dim))
        self._b2 = self.model.add_parameters(out_dim)
        self.epochs = epochs
        self.idx2labels = dict(enumerate(labels))
        self.labels2idx = {lab:ind for ind,lab in enumerate(labels)}
        self.useDropout=True
        #random.seed(1)

    def hid_layer(self,x,dropout):
        if dropout:
            W = dy.dropout(dy.parameter(self._W1),0.3)
            b = dy.dropout(dy.parameter(self._b1),0.3)
        else:
            W = dy.parameter(self._W1)
            b = dy.parameter(self._b1)
        return dy.rectify(W*x+b)

    def out_layer(self,x,dropout):
        if dropout:
            W = dy.dropout(dy.parameter(self._W2),0.3)
            b = dy.dropout(dy.parameter(self._b2),0.3)
        else:
            W = dy.parameter(self._W2)
            b = dy.parameter(self._b2)
        return dy.rectify(W*x+b)

    def predict_labels_softmax(self,x,dropout):
        x = dy.inputVector(x)
        h = self.hid_layer(x,dropout)
        y = self.out_layer(h,dropout)
        return dy.softmax(y)

    def predict(self,x_list):
        y = []
        for x in x_list:
            softmax_y = self.predict_labels_softmax(x,False)
            y_label = self.idx2labels[np.argmax(softmax_y.npvalue())]
            y.append(y_label)
        return y


    def train(self,data):
        prev_loss = float('inf')
        i = 0.
        loss_not_improving = 0
        while True:
            i += 1
            errors = 0.
            random.shuffle(data)
            losses = []
            total_loss = 0.
            for x,y in data:
                y_pred = self.predict_labels_softmax(x,self.useDropout)
                eloss = dy.pickneglogsoftmax(y_pred,self.labels2idx[y])
                total_loss += eloss.npvalue()
                losses.append(eloss)
                if y != self.idx2labels[np.argmax(y_pred.npvalue())]:
                    errors += 1
                if len(losses) > 100:
                    batch_loss = dy.esum(losses)
                    batch_loss.forward()
                    batch_loss.backward()
                    self.trainer.update()
                    dy.renew_cg()
                    losses = []

            errors = (errors/len(data))*100

            if len(losses)>0:
                batch_loss= dy.esum(losses)
                batch_loss.forward()
                batch_loss.backward()
                self.trainer.update()
                dy.renew_cg()

            if i%10==0:
                print "Epoch: %d Loss: %.3f - Error: %.1f"%(i, total_loss, errors)

            if prev_loss < total_loss:
                loss_not_improving += 1
            if loss_not_improving > 10 or i > self.epochs:
                break
            prev_loss = total_loss
