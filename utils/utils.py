import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import os
import itertools
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from PIL import Image


ce_loss = nn.CrossEntropyLoss()

def tr_acc(model, num_samples, train_loader):
    correct_num = 0
    train_loss = 0
    device = next(model.parameters()).device
    with torch.no_grad():
      for ind, (image_batch, label_batch) in enumerate(train_loader):
          
          image_batch = image_batch.to(device)
          label_batch = label_batch.to(device)
          # image_batch = image_batch.float()
          pred_array = model(image_batch)
          loss = ce_loss(pred_array, label_batch.long())
          prob, idx = torch.max(pred_array, dim=1)
          train_loss = train_loss + loss.cpu().data.numpy()*image_batch.shape[0]
          correct_num = correct_num + torch.eq(idx, label_batch).float().sum().cpu().numpy()
    return correct_num / num_samples, train_loss/num_samples

def test_batch(model, image, index, BATCH_SIZE,  nTrain_perClass, nvalid_perClass, halfsize):
    device = next(model.parameters()).device
    ind = index[0][nTrain_perClass[0]+ nvalid_perClass[0]:,:]
    nclass = len(index)
    true_label = np.zeros(ind.shape[0], dtype = np.int32)
    for i in range(1, nclass):
        ddd = index[i][nTrain_perClass[i] + nvalid_perClass[i]:,:]
        ind = np.concatenate((ind, ddd), axis = 0)
        tr_label = np.ones(ddd.shape[0], dtype = np.int32) * i
        true_label = np.concatenate((true_label, tr_label), axis = 0)
    length = ind.shape[0]
    if length % BATCH_SIZE != 0:
        add_num = BATCH_SIZE - length % BATCH_SIZE
        ff = range(length)
        add_ind = np.random.choice(ff, add_num, replace = False)
        add_ind = ind[add_ind]
        ind = np.concatenate((ind,add_ind), axis =0)
    test_label = np.zeros(ind.shape[0], dtype = np.int32)
    n = ind.shape[0] // BATCH_SIZE
    windowsize = 2 * halfsize + 1
    image_batch = np.zeros([BATCH_SIZE, windowsize, windowsize, image.shape[2]], dtype=np.float32)
    for i in range(n):
        for j in range(BATCH_SIZE):
            m = ind[BATCH_SIZE*i+j, :]
            image_batch[j,:,:,:] = image[(m[0] - halfsize):(m[0] + halfsize + 1),
                                                   (m[1] - halfsize):(m[1] + halfsize + 1),:]
        image_b = np.transpose(image_batch,(0,3,1,2))
        pred_array = model(torch.tensor(image_b).to(device))
        pred_label = torch.max(pred_array, dim=1)[1].cpu().data.numpy()
        test_label[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = pred_label
    predict_label = test_label[range(length)]

    confusion_matrix = metrics.confusion_matrix(true_label, predict_label)
    overall_accuracy = metrics.accuracy_score(true_label, predict_label)

    true_cla = np.zeros(nclass,  dtype=np.int64)
    for i in range(nclass):
        true_cla[i] = confusion_matrix[i,i]
    test_num_class = np.sum(confusion_matrix,1)
    test_num = np.sum(test_num_class)
    num1 = np.sum(confusion_matrix,0)
    po = overall_accuracy
    pe = np.sum(test_num_class*num1)/(test_num*test_num)
    kappa = (po-pe)/(1-pe)*100
    true_cla = np.true_divide(true_cla,test_num_class)*100
    average_accuracy = np.average(true_cla)
    print('overall_accuracy: {0:f}'.format(overall_accuracy*100))
    print('average_accuracy: {0:f}'.format(average_accuracy))
    print('kappa:{0:f}'.format(kappa))
    return true_cla, overall_accuracy*100, average_accuracy, kappa, confusion_matrix, predict_label

def spiral_scan_index(images):#### input: (B, C, H, W)   output: (B, C, H*W)
    height, width, channels = images.size()
    output = torch.zeros([height * width, channels],dtype=torch.long)
    direction = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = torch.zeros(height, width)

    x, y = 0, 0
    dx, dy = 0, 1

    for i in range(height * width):
        output[i] = images[x, y]
        visited[x, y] = 1
        next_x, next_y = x + dx, y + dy

        if 0 <= next_x < height and 0 <= next_y < width and visited[next_x, next_y] == 0:
            x, y = next_x, next_y
        else:
            dx, dy = direction[(direction.index((dx, dy)) + 1) % 4]
            x, y = x + dx, y + dy

    return output

def generate_matrix(H, W):
    # 生成坐标网格
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    A = torch.stack((grid_y, grid_x), dim=2)
    return A

def spiral_flatten(images):#### input: (B, C, H, W)   output: (B, C, H*W)
    B, C, H, W = images.size()
    index = generate_matrix(H, W)
    index = spiral_scan_index(index)

    indices = index[:, 0].view(-1, 1, 1), index[:, 1].view(-1, 1, 1)
    image_list = images[:, :, indices[0], indices[1]]
    image_list = image_list.squeeze(-1).squeeze(-1)
    return image_list

def s_flatten(tensor):
    B, C, H, W = tensor.size()
    reshaped_tensor = tensor.view(B, C, -1)

    for i in range(C):
        if i % 2 != 0:
            reshaped_tensor[:, i] = torch.flip(reshaped_tensor[:, i], [1])

    return reshaped_tensor

def record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, cm, training_time_ae, testing_time_ae, path):
    f = open(path, 'a')

    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n' + '\n'
    f.write(sentence5)
    sentence6 = 'Total average Training time is: ' + str(np.mean(training_time_ae)) + '\n'
    f.write(sentence6)
    sentence7 = 'Total average Testing time is: ' + str(np.mean(testing_time_ae)) + '\n' + '\n'
    f.write(sentence7)

    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    sentence8 = "Mean of all elements accuracy: " + str(element_mean) + '\n'
    f.write(sentence8)
    sentence9 = "Standard deviation of all elements accuracy: " + str(element_std) + '\n'
    f.write(sentence9)

    f.write("Mean of confusion matrix: " +'\n')
    cm = np.array(cm)
    mean_cm = np.mean(cm, axis = 0)
    for i in range(mean_cm.shape[0]):
        f.write(str(mean_cm[i]) + '\n')
    f.write("########################################################################################################" +'\n'+ '\n')
    f.close()

def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    """
    Compensate one for the stop value of range(...). because this function does not include the stop value.
    Two examples are listed as follows.
    When step = 1, supposing w = h = 3, W = H = 7, and step = 1.
    Then offset_w = 0, offset_h = 0.
    In this case, the x should have been ranged from 0 to 4 (4-6 is the last window),
    i.e., x is in range(0, 5) while W (7) - w (3) + offset_w (0) + 1 = 5. Plus one !
    Range(0, 5, 1) equals [0, 1, 2, 3, 4].

    When step = 2, supposing w = h = 3, W = H = 8, and step = 2.
    Then offset_w = 1, offset_h = 1.
    In this case, x is in [0, 2, 4] while W (8) - w (3) + offset_w (1) + 1 = 6. Plus one !
    Range(0, 6, 2) equals [0, 2, 4]/

    Same reason to H, h, offset_h, and y.
    """
    for x in range(0, W - w + offset_w + 1, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h + 1, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h



def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)

def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool_)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    print(f"target list: {target}")
    print(f"prediction list: {prediction}")
    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    return results



def convert_to_color(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def save_predictions(pred, gt=None, model_name="", caption=""):
    results_dir = './results/' + model_name
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if gt is None:
        image = Image.fromarray(pred)
        path = os.path.join(results_dir, caption)
        image.save(path)
    else:
        pred = Image.fromarray(pred)
        pred_path = os.path.join(results_dir, caption)
        pred.save(pred_path)

        gt = Image.fromarray(gt)
        gt_path = os.path.join(results_dir, f"gt_{caption}")
        gt.save(gt_path)


def show_results(results, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1 scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Aggregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1 scores"]
        kappa = results["Kappa"]

            
    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    # Calculate and display average accuracy per class from the confusion matrix
    if label_values is not None:
        class_sums = np.sum(cm, axis=1)
        valid_classes = class_sums != 0  # Identify classes with at least one prediction
        class_accuracies = np.diag(cm) / np.where(class_sums > 0, class_sums, np.nan)
        average_accuracy = np.nanmean(class_accuracies)  # Safely compute mean, ignoring NaN values
        text += "Average Accuracy: {:.05f}%\n".format(average_accuracy * 100)
    text += "---\n"

    if agregated:
        text += ("Overall Accuracy: {:.05f} +- {:.05f}\n".format(np.mean(accuracies),
                                                                 np.std(accuracies)))
    else:
        text += "Overall Accuracy : {:.05f}%\n".format(accuracy)
    text += "---\n"

    text += "F1 scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.05f} +- {:.05f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.05f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.05f} +- {:.05f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.05f}\n".format(kappa)

    print(text)

