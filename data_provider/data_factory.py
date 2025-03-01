from data_provider.data_loader import Dataset_Custom, Dataset_Pred, Dataset_Simulation, Dataset_Multi_Files, Dataset_Single_File, Dataset_OJ
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'multi': Dataset_Multi_Files,
    'single': Dataset_Single_File,
    'oj': Dataset_OJ
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.detail_freq
        Data = Dataset_Pred
    elif flag == 'simulation':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.detail_freq
        Data = Dataset_Simulation
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'multi':
        data_set = Dataset_Multi_Files(
            root_path=args.root_path,
            size=[args.seq_len, args.label_len, args.pred_len],
            data_prefix=args.data_prefix + '-' + flag,
            target=args.target,
            freq=freq,
        )
    elif args.data == 'single':
        if flag == 'train':
            data_set = Dataset_Single_File(
                root_path=args.root_path,
                file_name=args.data_prefix + '-' + flag + '.pkl',
                size=[args.seq_len, args.label_len, args.pred_len],
                target=args.target,
                count_target=args.count_target,
                zeros_pct=args.zeros_pct,
                selected_data_src=args.data_prefix + '-' + flag + '_selected.pkl',
                freq=freq,
                features=args.features
            )
        elif flag == 'val':
            data_set = Dataset_Single_File(
                root_path=args.root_path,
                file_name=args.data_prefix + '-' + flag + '.pkl',
                size=[args.seq_len, args.label_len, args.pred_len],
                target=args.target,
                count_target=args.count_target,
                zeros_pct=args.zeros_pct,
                selected_data_src=args.data_prefix + '-' + flag + '_selected.pkl',
                freq=freq,
                features=args.features
            )
    elif args.data == 'oj':
        if flag == 'train':
            data_set = Dataset_OJ(
                data_path=args.data_path,
                scale=True,
                subset_start=0, # 0
                subset_end=250000000, # 15000000
                seq_len=args.seq_len
            )
        elif flag == 'val':
            data_set = Dataset_OJ(
                data_path=args.data_path,
                scale=False,
                subset_start=650000000, # 15000000
                subset_end=700000000, # 22500000
                seq_len=args.seq_len
            )
        else:
            data_set = Dataset_OJ(
                data_path=args.data_path,
                scale=False,
                subset_start=700000000, # 22500000
                subset_end=700010000, # 30000000, 750000000
                seq_len=args.seq_len
            )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
