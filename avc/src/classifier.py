import numpy as np
import sklearn
from sklearn.metrics import accuracy_score

class Classifier(object):
    def __init__(self,task_type='transitivity',classifier_type='mlp'):
        self.task_type = task_type
        self.task_undefined = False
        self.classifier_type=classifier_type

    def get_xys(self,x_dict,y_dict):
        x_list = []
        y_list = []
        x_ids = []
        for key in x_dict:
            y_avc = y_dict[key[:2]]
            yItem = self.get_task_y(y_avc, self.task_type)
            if yItem is not None:
                y_list.append(yItem)
                x_list.append(x_dict[key])
                x_ids.append(key)
        return x_ids, x_list, y_list

    def get_y_gold(self,x_ids,y_gold):
        return [self.get_task_y(y_gold[x_id[:2]],self.task_type) for x_id in x_ids]

    def get_task_y(self,avc,task_type):
        #this should not be here or class misnamed
        if task_type == "transitivity":
            return sum([avc.has_dobj,avc.has_iobj])
        if task_type == "intransitive":
            return sum([avc.has_dobj,avc.has_iobj]) == 0
        elif task_type == "dobj":
            return int(avc.has_dobj)
        elif task_type == "iobj":
            return int(avc.has_iobj)
        elif task_type == "subj":
            return int(avc.has_subj)
        elif task_type == "subj_num":
            return avc.subj_num
        elif task_type == "subj_pers":
            return avc.subj_pers
        elif task_type == 'subj_n_pers':
            if avc.subj_num == None and avc.subj_pers == None:
                return None
            else:
                return "".join([str(i) for i in filter(None,[avc.subj_num,
                                                             avc.subj_pers])])
        else:
            raise Exception("Task unknown: %s"%task_type)

    def get_train_info(self,x_train, y_train):
        #check that we have at least 2 classes and 10 training examples
        if len(set(y_train)) < 2 or len(x_train) < 10:
            self.task_undefined = True
            self.trainsize = 0
        else:
            self.trainsize = len(x_train)

    def train(self,train_x,train_y):
        x_ids,x_train, y_train = self.get_xys(train_x,train_y)
        self.get_train_info(x_train, y_train)
        if self.task_undefined:
            return


        if self.classifier_type == 'perceptron':
            #averaged perceptron
            from sklearn.linear_model import SGDClassifier
            self.model = SGDClassifier(loss="perceptron",
                                       eta0=1, learning_rate="constant",
                                       penalty=None,average=10)
            self.model.fit(x_train, y_train)

        elif self.classifier_type == 'mlp':
            from multilayer_perceptron import MLP
            data = zip(x_train,y_train)
            labels = set(y_train)
            input_size = len(x_train[0])
            out_size = len(labels)
            hidden_size = input_size # same as Adi et al.
            self.model = MLP(input_size,hidden_size,out_size,labels,epochs=100)
            self.model.train(data)

    def predict(self,test_x,test_y):
        if not self.task_undefined:
            x_ids, x_test, y_test = self.get_xys(test_x,test_y)
            y_pred = self.model.predict(x_test)
            self.testsize = len(x_test)
            #write pred to file
            return x_ids,y_pred
        else:
            self.testsize = 0
            return None, None

    def evaluate(self,x_ids,pred,all_y_gold):
        if x_ids is not None and pred is not None:
            y_gold = self.get_y_gold(x_ids,all_y_gold)
            return accuracy_score(y_gold,pred)
        else:
            return np.nan

    def majority_baseline(self,train_x,train_y,test_x,test_y):
        x_ids,x_train, y_train = self.get_xys(train_x,train_y)
        x_ids,x_test, y_test = self.get_xys(test_x,test_y)
        y_maj = max(y_train, key=y_train.count)
        y_maj_pred = [y_maj for i in range(len(y_test))]
        return accuracy_score(y_test, y_maj_pred)

