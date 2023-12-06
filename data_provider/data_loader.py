import pickle

import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Data_Fragment_Info:
    def __init__(self, file_name, num_training_data, num_lines, start_index, end_index):
        self.file_name = file_name
        self.num_training_data = num_training_data
        self.num_lines = num_lines
        self.start_index = start_index
        self.end_index = end_index


class Dataset_Single_File(Dataset):
    def __init__(self, root_path, file_name, size, target, count_target, zeros_pct=1.0, selected_data_src=None, scale=True, freq='u', features='MR'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.target = target
        self.count_target = count_target
        self.scale = scale
        self.count_target_zero_scaled = 0
        self.zero_scaled_input = None
        self.zeros_pct = zeros_pct
        self.freq = freq
        self.features = features

        self.root_path = root_path
        self.file_name = file_name
        self.target_index = None
        self.count_target_index = None
        self.training_indices = []
        self.selected_data = []
        self.__read_data__()

        if zeros_pct < 1.0 and selected_data_src is None:
            self.__count__()
            # save date_set.selected_data to a file
            with open(os.path.join(self.root_path, self.file_name[:-4] + '_selected.pkl'), 'wb') as f:
                pickle.dump(self.selected_data, f)

        if selected_data_src is not None:
            self.__read_selected_data__(selected_data_src),

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.data = pd.read_pickle(os.path.join(self.root_path, self.file_name))
        self.data = self.data.drop(self.data.filter(regex='price').columns, axis=1)
        self.target_index = self.data.columns.get_loc(self.target)
        self.count_target_index = self.data.columns.get_loc(self.count_target)
        self.training_indices = [i for i in range(self.data.shape[1]) if i != self.target_index]
        self.data_stamp = time_features(self.data.index, freq=self.freq)
        self.data_stamp = self.data_stamp.transpose(1, 0)
        if self.scale:
            self.scaler.fit(self.data.values)
            self.data = self.scaler.transform(self.data.values)
            # create zero matrix with the same dimensions as the self.data
            zero_matrix = np.zeros(self.data.shape)
            self.zero_scaled_input = self.scaler.transform(zero_matrix)[0]
            self.count_target_zero_scaled = self.zero_scaled_input[self.count_target_index]

    def __read_selected_data__(self, selected_data_src):
        path = os.path.join(self.root_path, selected_data_src)
        with open(path, 'rb') as data:
            self.selected_data = pickle.load(data)
        print(f'selected = {len(self.selected_data)}')
        print(f'total = {len(self.data)}')
        print(f'% = {len(self.selected_data)/len(self.data)}')


    def __count__(self):
        num_data = len(self.data) - self.seq_len - self.pred_len + 1
        zero_array = np.full(self.seq_len, self.count_target_zero_scaled)
        for i in range(num_data):
            seq_x, _ = self.__get_x_y__(i, self.count_target_index)

            # check if seq_x only contains zeros
            seq_x_zero = np.isclose(seq_x, zero_array).all()

            # if seq_x_zero is True, then zero_pct of the time add i to selected_data
            if seq_x_zero:
                if np.random.rand() < self.zeros_pct:
                    self.selected_data.append(i)
            else:
                self.selected_data.append(i)

            # print every 1% of the way through the data
            if i % (num_data // 100) == 0:
                print(f'{i / (num_data // 100)}%')

    def __get_x_y__(self, index, selected_col_idx=None):
        s_end = index + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        if selected_col_idx is None:
            seq_x = self.data[index:s_end]
            seq_y = self.data[r_begin:r_end]
        else:
            seq_x = self.data[index:s_end, selected_col_idx]
            seq_y = self.data[r_begin:r_end, selected_col_idx]
        return seq_x, seq_y

    def __get_x_y_regression__(self, index):
        s_end = index + self.seq_len
        # self.data is a numpy ndarray. Select all columns except the target column with index self.target_index
        seq_x = self.data[index:s_end, self.training_indices]
        seq_y = self.data[s_end - 1, self.target_index]
        return seq_x, seq_y

    def get_zero_scaled(self, feature_idx):
        return self.zero_scaled_input[feature_idx]

    def __len__(self):
        if self.zeros_pct < 1.0:
            return len(self.selected_data)
        else:
            return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        i = index
        if self.zeros_pct < 1.0:
            i = self.selected_data[i]
        if self.features == 'MR':
            seq_x, seq_y = self.__get_x_y_regression__(i)
        else:
            seq_x, seq_y = self.__get_x_y__(i)
        return seq_x, seq_y


class Dataset_Multi_Files2(Dataset):
    def __init__(self, root_path, size, data_prefix, target, scale=True, freq='u'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.target = target
        self.scale = scale
        self.freq = freq

        self.root_path = root_path
        self.data_prefix = data_prefix
        self.current_fragment = 0
        self.data_fragments = []
        self.total_len = 0

        self.__init()
        self.__read_data__()


class Dataset_Multi_Files(Dataset):
    def __init__(self, root_path, size, data_prefix, target, scale=True, freq='u'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.target = target
        self.scale = scale
        self.freq = freq

        self.root_path = root_path
        self.data_prefix = data_prefix
        self.current_fragment = 0
        self.data_fragments = []
        self.total_len = 0

        self.__init()
        self.__read_data__()

    def __init(self):
        for file in os.listdir(self.root_path):
            # if file does not contain _selected, then it is a data file
            if file.startswith(self.data_prefix) and not file.endswith('_selected.pkl'):
                df_raw = pd.read_pickle(os.path.join(self.root_path, file))
                num_data = len(df_raw) - self.seq_len - self.pred_len + 1
                data_fragment_info = Data_Fragment_Info(file, num_data, len(df_raw), self.total_len, self.total_len + num_data)
                self.data_fragments.append(data_fragment_info)
                self.total_len += num_data

        self.scaler = StandardScaler()

    def __read_data__(self):
        fragment = self.data_fragments[self.current_fragment]
        self.data = pd.read_pickle(os.path.join(self.root_path, fragment.file_name))
        self.data_stamp = time_features(self.data.index, freq=self.freq)
        self.data_stamp = self.data_stamp.transpose(1, 0)
        if self.scale:
            self.scaler.fit(self.data.values)
            self.data = self.scaler.transform(self.data.values)

    def __select_data_fragment(self, index):
        current_max_index = 0
        current_file_index = 0
        for fragment in self.data_fragments:
            current_max_index += fragment.num_training_data
            if index <= current_max_index:
                break
            else:
                current_file_index += 1
        if current_file_index != self.current_fragment:
            self.current_fragment = current_file_index
            self.__read_data__()

    def __getitem__(self, index):
        self.__select_data_fragment(index)
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.total_len

    def inverse_transform(self, seq_x, seq_y):
        # self.scaler.fit(seq_x)
        return self.scaler.inverse_transform(seq_y)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Simulation(Dataset):
    def __init__(self, root_path, flag='simulation', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        border1 = 0
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# main
if __name__ == '__main__':
    root_path = '../dataset/ticker/'
    file_name = 'seven-days-train.pkl'
    data_set = Dataset_Single_File(
        root_path=root_path,
        file_name=file_name,
        size=[128, 64, 128],
        target='log_return_s',
        zeros_pct=0.03,
        scale=True
    )
    print(len(data_set))
    print(len(data_set) / (len(data_set.data) - data_set.seq_len - data_set.pred_len + 1))
    # save date_set.selected_data to a file
    with open('../dataset/ticker/seven-days-train_selected.pkl', 'wb') as f:
        pickle.dump(data_set.selected_data, f)
