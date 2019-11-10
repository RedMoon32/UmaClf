import requests
import numpy as np
import zipfile
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import cv2
import os
import logging
from constants import *
from network import *


def get_logger(name='Bot-Classiffier'):
    """Get Logger for Console Outputting"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    return logger


logger = get_logger()


class FootballersModel:
    """ Full data preprocessing pipeline"""

    def __init__(self):
        logger.info("Initiating Model")
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        #logger.info("Downloading data")
        # self._download_data()

        x, y, self.label_mapping = self._get_train_data()
        self.target_image_size = self._get_target_image_size(x)

        if os.path.exists(out_model_path):
            logger.info("Existing model found - Loading trained model")
            self.model = Net().to(self.device)
            self.model.load_state_dict(torch.load(out_model_path, map_location=self.device))
            self.model.eval()
        else:
            logger.info("Existing model not found\nTraining model")
            x = self._preprocess(x)
            train, test = self._get_train_valid_split(x, y)
            self.model = self._train(train, test)

        logger.info("Model was initiated")

    def predict(self, inp_):
        """ Predict output by input image and return class name"""
        inp = inp_.copy()
        img = self._preproc_img(inp)
        out = self.model(torch.Tensor(np.array([img])).to(self.device))
        out = out.argmax(dim=1, keepdim=True).cpu().numpy()[0][0]
        return self.label_mapping[out]

    def _download_data(self):
        """ Download and Unzip train data"""
        myfile = requests.get(url)
        open(zip_f, 'wb').write(myfile.content)
        with zipfile.ZipFile(zip_f, 'r') as zip_ref:
            zip_ref.extractall(target_path)

    def _get_train_data(self):
        """ Read image folder and return np arrays for data and labels"""
        labels_out = {}
        x = []
        y = []

        with open(os.path.join(target_path, csv_inside_path), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                x.append(np.asarray(np.array(plt.imread(target_path + '/images/' + row[0] + '.png'), np.float)))
                y.append(int(row[1]))
                labels_out[int(row[1])] = row[2]

        x = np.array(x)
        y = np.array(y)

        return x, y, labels_out

    def _get_target_image_size(self, x):
        """ Get image size as standart"""
        shapes = []
        for im in x:
            shapes.append(im.shape)
        shapes = np.array(shapes)
        h_size, w_size = int(np.average(shapes[:, 0])), int(np.average(shapes[:, 1]))
        return h_size, w_size

    def _preproc_img(self, x):
        """ Preprocess single Image"""
        x = cv2.resize(x, self.target_image_size)
        x = x.transpose((2, 0, 1))
        x = x.astype(int)
        return x

    def _preprocess(self, x):
        """ Preprocess full Array"""
        x *= 255
        for ind in range(x.shape[0]):
            x[ind] = self._preproc_img(x[ind])
        return np.array(x.tolist(), 'int64')

    def _create_dataset(self, x, y):
        """ Create Tensor Dataset for train loader"""
        tensor_x = torch.Tensor(x)
        tensor_y = torch.Tensor(y).long()
        dataset = TensorDataset(tensor_x, tensor_y)
        return dataset

    def _get_train_valid_split(self, x, y):

        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_split, random_state=seed, stratify=y)

        train_dataset = self._create_dataset(X_train, Y_train)
        test_dataset = self._create_dataset(X_test, Y_test)

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
        return train_loader, test_loader

    def _train(self, train_loader, test_loader):
        """ Train new model by train_loader and test_loader"""
        model = Net().to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            train(model, self.device, train_loader, optimizer, epoch)
            test('Footballers test set', model, self.device, test_loader, logger)

        torch.save(model.state_dict(), out_model_path)
        return model
