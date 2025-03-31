import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from datetime import timedelta
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import os


torch.cuda.manual_seed()
loss_func = torch.nn.CrossEntropyLoss()
learning_rate = 2e-5
num_epochs = 200
require_improvement = 2000

DEVICE = torch.device("cuda:0")
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(model, train_iter, val_iter, test_iter,class_sentiment,class_intention,class_offensiveness,class_m_occurrence,class_m_category,is_meta):
    if is_meta == 'both':
        num = 5
    else:
        num = 3

    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.1)
    total_batch = 0
    dev_best_loss_list = []
    for i in range(num):
        dev_best_loss_list.append(float('inf'))

    last_improve_list = []
    for i in range(num):
        last_improve_list.append(0)
    flag = False
    for epoch in range(num_epochs):
        total_sentiment_acc = 0
        total_intention_acc = 0
        total_offensiveness_acc = 0
        total_m_occurrence_acc = 0

        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            text = trains[0].to(DEVICE)
            meta = trains[1].to(DEVICE)
            meta_pad = trains[2].to(DEVICE)
            image = trains[3].to(DEVICE)
            id = trains[4]
            source = trains[5].to(DEVICE)
            target = trains[6].to(DEVICE)
            text_mate = trains[7].to(DEVICE)
            label_sentiment = labels[0].to(DEVICE)
            label_intention = labels[1].to(DEVICE)
            label_offensiveness = labels[2].to(DEVICE)
            label_m_occurrence = labels[3].to(DEVICE)
            label_m_category = labels[4].to(DEVICE)
            outputs_sentiment, outputs_intention, outputs_offensiveness, outputs_m_occurrence, outputs_m_occurrence_real, loss_dg = model(
                text, meta, meta_pad, image, source, target, text_mate)

            model.zero_grad()

            loss_1 = loss_func(outputs_sentiment, label_sentiment.long())
            loss_2 = loss_func(outputs_intention, label_intention.long())
            loss_3 = loss_func(outputs_offensiveness, label_offensiveness.long())
            loss_4 = loss_func(outputs_m_occurrence, label_m_occurrence.long())
            loss_5 = loss_func(outputs_m_occurrence_real, label_m_occurrence.long())

            loss_list = [loss_1, loss_2, loss_3, loss_4, loss_5, loss_dg]
            loss = loss_1+loss_2+loss_3

            loss_4.backward(retain_graph=True)
            loss_5.backward(retain_graph=True)
            loss.backward(retain_graph=True)
            loss_dg.backward(retain_graph=True)
            optimizer.step()

            if total_batch % 100 == 0:
                _, predic_sentiment = outputs_sentiment.cpu().max(1)
                _, predic_intention = outputs_intention.cpu().max(1)
                _, predic_offensiveness = outputs_offensiveness.cpu().max(1)
                _, predic_m_occurrence = outputs_m_occurrence.cpu().max(1)
                _, predic_m_occurrence_real = outputs_m_occurrence_real.cpu().max(1)

                train_acc_sentiment = metrics.accuracy_score(label_sentiment.cpu(), predic_sentiment)
                train_acc_intention = metrics.accuracy_score(label_intention.cpu(), predic_intention)
                train_acc_offensiveness = metrics.accuracy_score(label_offensiveness.cpu(), predic_offensiveness)
                train_acc_m_occurrence = metrics.accuracy_score(label_m_occurrence.cpu(), predic_m_occurrence)
                train_acc_m_occurrence_real = metrics.accuracy_score(label_m_occurrence.cpu(), predic_m_occurrence_real)


                train_acc_list = [train_acc_sentiment,train_acc_intention,train_acc_offensiveness,train_acc_m_occurrence,train_acc_m_occurrence_real]

                dev_acc_list,dev_pre_list,dev_rec_list, dev_loss_list = evaluate(model, val_iter,
                                                                 class_sentiment=class_sentiment,
                                                                 class_intention=class_intention,
                                                                 class_offensiveness=class_offensiveness,
                                                                 class_m_occurrence=class_m_occurrence,is_meta=is_meta)

                for j in range(num):
                    if dev_loss_list[j] < dev_best_loss_list[j]:
                        dev_best_loss_list[j] = dev_loss_list[j]

                test(model, val_iter, val=True, class_sentiment=class_sentiment,
                     class_intention=class_intention,
                     class_offensiveness=class_offensiveness,
                     class_m_occurrence=class_m_occurrence, is_meta=is_meta)

                test(model, test_iter, val=False, class_sentiment=class_sentiment,
                     class_intention=class_intention,
                     class_offensiveness=class_offensiveness,
                     class_m_occurrence=class_m_occurrence, is_meta=is_meta)


                time_dif = get_time_dif(start_time)
                msg_list = ['Sentiment: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} ',
                            'Intention: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} ',
                            'Offensiveness: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5}',
                            'Metaphor_occurrence: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} ',
                            'Metaphor_occurrence Real: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} ',
                            ]
                for j in range(num):
                    print(msg_list[j].format(total_batch, loss_list[j].item(), train_acc_list[j], dev_loss_list[j], dev_acc_list[j], time_dif))
                model.train()
            total_batch += 1


    test(model, test_iter, val=False,class_sentiment=class_sentiment,
                                                                     class_intention=class_intention,
                                                                     class_offensiveness=class_offensiveness,
                                                                     class_m_occurrence=class_m_occurrence,is_meta=is_meta)

def test(model, test_iter, val=False,class_sentiment=7,class_intention=5,class_offensiveness=4,class_m_occurrence=2,class_m_category=3,is_meta=None):

    model.eval()
    start_time = time.time()
    test_acc_list,test_pre_list,test_rec_list, test_loss_list, test_report_list, test_confusion_list = evaluate(model, test_iter, test=True,
                                                                class_sentiment=class_sentiment,
                                                                 class_intention=class_intention,
                                                                 class_offensiveness=class_offensiveness,
                                                                 class_m_occurrence=class_m_occurrence,is_meta=is_meta)
    if val:
        msg_list = ['Sentiment: Val Loss: {0:>5.2},  Val Acc: {1:>6.2%}, Val Precision: {2:>6.2%},  Val Recall: {3:>6.2%}\n',
                    'Intention: Val Loss: {0:>5.2},  Val Acc: {1:>6.2%}, Val Precision: {2:>6.2%},  Val Recall: {3:>6.2%}\n',
                    'Offensiveness: Val Loss: {0:>5.2},  Val Acc: {1:>6.2%}, Val Precision: {2:>6.2%},  Val Recall: {3:>6.2%}\n',
                    'Metaphor_occurrence: Val Loss: {0:>5.2},  Val Acc: {1:>6.2%}, Val Precision: {2:>6.2%},  Val Recall: {3:>6.2%}\n',
                    'Metaphor_occurrence Real: Val Loss: {0:>5.2},  Val Acc: {1:>6.2%}, Val Precision: {2:>6.2%},  Val Recall: {3:>6.2%}\n']
    else:
        msg_list = ['Sentiment: Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test Precision: {2:>6.2%},  Test Recall: {3:>6.2%}\n',
                    'Intention: Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test Precision: {2:>6.2%},  Test Recall: {3:>6.2%}\n',
                    'Offensiveness: Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test Precision: {2:>6.2%},  Test Recall: {3:>6.2%}\n',
                    'Metaphor_occurrence: Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test Precision: {2:>6.2%},  Test Recall: {3:>6.2%}\n',
                    'Metaphor_occurrence Real: Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test Precision: {2:>6.2%},  Test Recall: {3:>6.2%}\n']

    if is_meta == 'both':
        num = 5
    else:
        num = 3

    for i in range(num):
        print(msg_list[i].format(test_loss_list[i], test_acc_list[i], test_pre_list[i], test_rec_list[i]))
        print("Precision, Recall and F1-Score...")
        print(test_report_list[i])
        print("Confusion Matrix...")
        print(test_confusion_list[i])
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    if val:
        with open('best_val_results.txt', 'a', encoding='utf-8') as f:
            for i in range(num):
                f.write(msg_list[i].format(test_loss_list[i], test_acc_list[i], test_pre_list[i], test_rec_list[i]))
                f.write("\nPrecision, Recall and F1-Score...\n")
                f.write(test_report_list[i])
                f.write("\nConfusion Matrix...\n")
                f.write(str(test_confusion_list[i]))
    else:
        with open('best_test_results.txt', 'a', encoding='utf-8') as f:
            for i in range(num):
                f.write(msg_list[i].format(test_loss_list[i], test_acc_list[i], test_pre_list[i], test_rec_list[i]))
                f.write("\nPrecision, Recall and F1-Score...\n")
                f.write(test_report_list[i])
                f.write("\nConfusion Matrix...\n")
                f.write(str(test_confusion_list[i]))

def evaluate(model, data_iter, test=False,class_sentiment=7,class_intention=5,class_offensiveness=4,class_m_occurrence=2,class_m_category=3,is_meta=None):
    model.eval()
    loss_total_1 = 0
    loss_total_2 = 0
    loss_total_3 = 0
    loss_total_4 = 0
    loss_total_5 = 0

    predict_all_1 = np.array([], dtype=int)
    predict_all_2 = np.array([], dtype=int)
    predict_all_3 = np.array([], dtype=int)
    predict_all_4 = np.array([], dtype=int)
    predict_all_5 = np.array([], dtype=int)

    labels_all_1 = np.array([], dtype=int)
    labels_all_2 = np.array([], dtype=int)
    labels_all_3 = np.array([], dtype=int)
    labels_all_4 = np.array([], dtype=int)
    labels_all_5 = np.array([], dtype=int)

    with torch.no_grad():
        for (tests, labels) in data_iter:
            text = tests[0].to(DEVICE)
            meta = tests[1].to(DEVICE)
            meta_pad = tests[2].to(DEVICE)

            image = tests[3].to(DEVICE)
            source =  tests[5].to(DEVICE)
            target =  tests[6].to(DEVICE)
            text_mate =  tests[7].to(DEVICE)

            label_sentiment = labels[0].to(DEVICE)
            label_intention = labels[1].to(DEVICE)
            label_offensiveness = labels[2].to(DEVICE)
            label_m_occurrence = labels[3].to(DEVICE)
            outputs_sentiment, outputs_intention, outputs_offensiveness, outputs_m_occurrence,outputs_m_occurrence_real, loss_dg = model(text, meta, meta_pad,image,source,target,text_mate)

            loss_1 = loss_func(outputs_sentiment, label_sentiment.long())
            loss_2 = loss_func(outputs_intention, label_intention.long())
            loss_3 = loss_func(outputs_offensiveness, label_offensiveness.long())
            loss_4 = loss_func(outputs_m_occurrence, label_m_occurrence.long())
            loss_5 = loss_func(outputs_m_occurrence_real,label_m_occurrence.long())
            loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_dg

            loss_total_1 += loss_1
            loss_total_2 += loss_2
            loss_total_3 += loss_3
            loss_total_4 += loss_4
            loss_total_5 += loss_5

            _, predict_sentiment = outputs_sentiment.cpu().max(1)
            _, predict_intention = outputs_intention.cpu().max(1)
            _, predict_offensiveness = outputs_offensiveness.cpu().max(1)
            _, predict_m_occurrence = outputs_m_occurrence.cpu().max(1)
            _, predict_m_occurrence_real = outputs_m_occurrence_real.cpu().max(1)

            labels_all_1 = np.append(labels_all_1, label_sentiment.cpu())
            labels_all_2 = np.append(labels_all_2, label_intention.cpu())
            labels_all_3 = np.append(labels_all_3, label_offensiveness.cpu())
            labels_all_4 = np.append(labels_all_4, label_m_occurrence.cpu())
            labels_all_5 = np.append(labels_all_5, label_m_occurrence.cpu())

            predict_all_1 = np.append(predict_all_1, predict_sentiment)
            predict_all_2 = np.append(predict_all_2, predict_intention)
            predict_all_3 = np.append(predict_all_3, predict_offensiveness)
            predict_all_4 = np.append(predict_all_4, predict_m_occurrence)
            predict_all_5 = np.append(predict_all_5, predict_m_occurrence_real)

    acc_1 = metrics.accuracy_score(labels_all_1, predict_all_1)
    acc_2 = metrics.accuracy_score(labels_all_2, predict_all_2)
    acc_3 = metrics.accuracy_score(labels_all_3, predict_all_3)
    acc_4 = metrics.accuracy_score(labels_all_4, predict_all_4)

    pre_1 = metrics.precision_score(labels_all_1, predict_all_1, average='weighted')
    pre_2 = metrics.precision_score(labels_all_2, predict_all_2, average='weighted')
    pre_3 = metrics.precision_score(labels_all_3, predict_all_3, average='weighted')
    pre_4 = metrics.precision_score(labels_all_4, predict_all_4, average='weighted')

    rec_1 = metrics.recall_score(labels_all_1, predict_all_1, average='weighted')
    rec_2 = metrics.recall_score(labels_all_2, predict_all_2, average='weighted')
    rec_3 = metrics.recall_score(labels_all_3, predict_all_3, average='weighted')
    rec_4 = metrics.recall_score(labels_all_4, predict_all_4, average='weighted')

    if is_meta == 'both':
        acc_5 = metrics.accuracy_score(labels_all_5, predict_all_5)
        pre_5 = metrics.precision_score(labels_all_5, predict_all_5, average='weighted')
        rec_5 = metrics.recall_score(labels_all_5, predict_all_5, average='weighted')
    if test:
        sentiment_list = [str(i+1) for i in range(class_sentiment)]
        intention_list = [str(i+1) for i in range(class_intention)]
        offensiveness_list = [str(i+1) for i in range(class_offensiveness)]
        m_occurrence_list = [str(i+1) for i in range(class_m_occurrence)]
        m_occurrence_real_list = [str(i+1) for i in range(class_m_occurrence)]

        report_sentiment = metrics.classification_report(labels_all_1, predict_all_1, target_names=sentiment_list,
                                                         digits=4)
        report_intention = metrics.classification_report(labels_all_2, predict_all_2, target_names=intention_list,
                                                         digits=4)
        report_offensiveness = metrics.classification_report(labels_all_3, predict_all_3, target_names=offensiveness_list,
                                                         digits=4)

        if is_meta == 'both':
            report_m_occurrence_real = metrics.classification_report(labels_all_5, predict_all_5, target_names=m_occurrence_real_list,
                                                         digits=4)
            report_m_occurrence = metrics.classification_report(labels_all_4, predict_all_4, target_names=m_occurrence_list,
                                                         digits=4)

        confusion_sentiment = metrics.confusion_matrix(labels_all_1, predict_all_1)
        confusion_intention = metrics.confusion_matrix(labels_all_2, predict_all_2)
        confusion_offensiveness = metrics.confusion_matrix(labels_all_3, predict_all_3)

        if is_meta == 'both':
            confusion_m_occurrence = metrics.confusion_matrix(labels_all_4, predict_all_4)
            confusion_m_occurrence_real = metrics.confusion_matrix(labels_all_5, predict_all_5)

        if is_meta == 'both':
            return [acc_1,acc_2,acc_3,acc_4,acc_5], \
                   [pre_1, pre_2, pre_3, pre_4, pre_5], \
                   [rec_1, rec_2, rec_3, rec_4, rec_5], \
                   [loss_total_1/len(data_iter),loss_total_2/len(data_iter),
                    loss_total_3/len(data_iter),loss_total_4/len(data_iter),loss_total_5/len(data_iter)],\
                   [report_sentiment,report_intention,report_offensiveness,report_m_occurrence,report_m_occurrence_real],\
                   [confusion_sentiment,confusion_intention,confusion_offensiveness,confusion_m_occurrence,confusion_m_occurrence_real]
        else:
            return [acc_1, acc_2, acc_3], \
                   [pre_1, pre_2, pre_3], \
                   [rec_1, rec_2, rec_3], \
                   [loss_total_1 / len(data_iter), loss_total_2 / len(data_iter),loss_total_3/len(data_iter)], \
                   [report_sentiment, report_intention, report_offensiveness], \
                   [confusion_sentiment, confusion_intention, confusion_offensiveness]

    if is_meta == 'both':
        return [acc_1,acc_2,acc_3,acc_4,acc_5], \
               [pre_1, pre_2, pre_3, pre_4, pre_5], \
               [rec_1, rec_2, rec_3, rec_4, rec_5], \
               [loss_total_1/len(data_iter),loss_total_2/len(data_iter),
                    loss_total_3/len(data_iter),loss_total_4/len(data_iter),loss_total_5/len(data_iter)]
    else:
        return [acc_1, acc_2, acc_3], \
                   [pre_1, pre_2, pre_3], \
                   [rec_1, rec_2, rec_3], \
               [loss_total_1 / len(data_iter), loss_total_2 / len(data_iter),
                loss_total_3 / len(data_iter)]
