import time
import datetime
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim

from config import load_args
from utils.data_read import *
from utils.utils import *
# from models.model import mamba_1D_model, mamba_2D_model, mamba_SS_model
# # from models.model_bfs import mamba_my_model
from models.model_spatial import mamba_spatial
from calflops import calculate_flops


day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')
args = load_args()

num_of_ex = 10
dataset = args.dataset
windowsize = args.windowsize
type = args.type


train_num = args.train_num
val_num = args.val_num
sample_mode = args.sample_mode


lr = args.lr
epoch = args.epoch
batch_size = args.batch_size
drop_rate = args.drop_rate
gamma = args.lr_decay

model_id = args.model_id

windowsize = args.windowsize
spe_windowsize = args.spe_windowsize
spa_patch_size = args.spa_patch_size
spe_patch_size = args.spe_patch_size
spa_depth = 4
spe_depth = 2
embed_dim = args.embed_dim
hid_chans = args.hid_chans
depth = args.depth
use_bi = args.use_bi
use_global = args.use_global
use_cls = args.use_cls
use_bfs = False

net_name_candidate = ['mamba_1D_model', 'mamba_2D_model', 'mamba_SS_model', 'mamba_my_model', 'mamba_spatial']
net_name = net_name_candidate[model_id]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ce_loss = torch.nn.CrossEntropyLoss()

@torch.no_grad()
def test(model):
    patch_size = windowsize
    center_pixel = args.center_pixel
    n_classes = np.max(label.astype(np.int32))+1

    probs_map = np.zeros(image.shape[:2] + (n_classes,))
    kwargs = {
        "step": 1,
        "window_size": (patch_size, patch_size),
    }
    iterations = count_sliding_window(image, **kwargs) // batch_size
    for batch in tqdm(
            grouper(batch_size, sliding_window(image, **kwargs)),
            total=(iterations),
            desc="Inference on the image",
    ):
        if patch_size == 1:
            data = [b[0][0, 0] for b in batch] 
            data = np.copy(data)
            data = torch.from_numpy(data)
        else:
            data = [b[0] for b in batch] 
            data = np.copy(data)
            data = data.transpose(0, 3, 1, 2)
            data = torch.from_numpy(data)
            data = data.unsqueeze(1)

        indices = [b[1:] for b in batch]
        data = data.cuda(non_blocking=True)
        output = model(data)

        if isinstance(output, tuple):
                output = output[0]

        output = output.to("cpu")
        if patch_size == 1 or center_pixel:
            output = output.numpy()
        else:
            output = np.transpose(output.numpy(), (0, 2, 3, 1))
        for (x, y, w, h), out in zip(indices, output):
            if center_pixel:
                probs_map[x + w // 2, y + h // 2] += out
            else:
                probs_map[x: x + w, y: y + h] += out
        
    return probs_map



results = []
for num in range(0, num_of_ex):
    print('num:', num)
    random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    np.random.seed(num)
    torch.backends.cudnn.deterministic = True
    
    image, label, palette, label_values = readdata(windowsize, args)
    train_label, test_label = sample_gt(label, train_num, num, sample_mode)
    # val_label, train_label = sample_gt(train_label, val_num, num, sample_mode)
    nclass = np.max(label.astype(np.int32))+1
    nband = image.shape[-1]

    train_dataset = HyperX(image, train_label, args)
    # val_dataset = HyperX(image, val_label, args)
    test_dataset = HyperX(image, test_label, args)
    
    data_loader_train, data_loader_val, data_loader_test = build_loader(train_dataset, None, test_dataset, batch_size)
    # if model_id == 0:
    #     model = mamba_1D_model(img_size=(spe_windowsize,spe_windowsize), spa_img_size=(windowsize, windowsize), nband=nband, patch_size=spe_patch_size, embed_dim=embed_dim, nclass=nclass, depth=depth, bi=use_bi, norm_layer=nn.LayerNorm, global_pool=use_global, cls = use_cls)
    # elif model_id == 1:
    #     model = mamba_2D_model(img_size=(windowsize, windowsize), patch_size=spa_patch_size, in_chans=nband, hid_chans = hid_chans, embed_dim=embed_dim, nclass=nclass, drop_path=drop_rate, depth=4, bi=use_bi, norm_layer=nn.LayerNorm, global_pool=use_global, cls = use_cls)
    # if model_id == 2:  
    #     model = mamba_SS_model(spa_img_size=(windowsize, windowsize),spe_img_size=(spe_windowsize,spe_windowsize), spa_patch_size=spa_patch_size, spe_patch_size=spe_patch_size, in_chans=nband, hid_chans = hid_chans, embed_dim=embed_dim, drop_path=drop_rate, nclass=nclass, depth=depth, bi=use_bi, norm_layer=nn.LayerNorm, global_pool=use_global, cls = use_cls, fu = args.use_fu, bfs=use_bfs)
    # elif model_id == 3:
    #     model = mamba_my_model(spa_img_size=(windowsize, windowsize),spa_patch_size=spa_patch_size, in_chans=nband, hid_chans = hid_chans, embed_dim=embed_dim, drop_path=drop_rate, nclass=nclass, spa_depth=spa_depth, spe_depth=spe_depth,
    #                            bi=use_bi, norm_layer=nn.LayerNorm, global_pool=use_global, cls = use_cls, bfs=use_bfs)
    if model_id == 4:
        model = mamba_spatial(spa_img_size=(windowsize, windowsize),spe_img_size=(spe_windowsize,spe_windowsize), spa_patch_size=spa_patch_size, spe_patch_size=spe_patch_size, in_chans=nband, hid_chans = hid_chans, embed_dim=embed_dim, drop_path=drop_rate, nclass=nclass, depth=depth, bi=use_bi, norm_layer=nn.LayerNorm, global_pool=use_global, cls = use_cls, fu = args.use_fu)
    else:
        raise Exception('model id does not find')
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr = lr, weight_decay = 1e-4)
    print('the number of training samples:', len(train_dataset))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [80, 140, 170], gamma = gamma, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,  T_0=5,T_mult=2)

    # print("Detailed network architecture:")
    # # print(model.__repr__())
    # total_params = 0
    # # for name, param in model.named_parameters():
    # #     if param.requires_grad:
    # #         num = param.numel() 
    # #         print(f"{name}: {num}")
    # #         total_params += num
    # n_parameters = sum(p.numel() for p in model.parameters()
    #                    if p.requires_grad)
    # print(f"Number of params: {n_parameters}")

    # calculate flops and parms
    # model.eval()
    # flops, macs1, para = calculate_flops(model=model,
    #                                     input_shape=(1, 1, image.shape[2], 27, 27), )
    # print("para:{}\n,flops:{}".format(para, flops))
    
    # training
    tic1 = time.time()
    for i in range(epoch):
        model.train()
        train_loss = 0
        for idx, (label_x, label_y) in enumerate(data_loader_train):
            label_x, label_y = label_x.to(device), label_y.to(device)
            # print(f"label_x shape: {label_x.shape}")
            outputs = model(label_x)
            loss = ce_loss(outputs, label_y.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()*label_x.shape[0]

        train_loss = train_loss/len(train_dataset)
        scheduler.step()
        
        if (i+1) % 5 == 0:
            train_acc, train_loss = tr_acc(model.eval(), len(train_dataset), data_loader_train)
            # val_acc, val_loss = tr_acc(model.eval(), len(val_dataset), data_loader_val)
            
            print('epoch:', i, 'loss:%.4f' % train_loss,'train_acc:%.4f'%train_acc.item())

    toc1 = time.time()
    probs_map = test(model.eval())
    toc2 = time.time()

    num_cls = int(probs_map.shape[2])
    prob_map = np.argmax(probs_map, axis=-1)
    result = metrics(prob_map, test_label, ignored_labels=args.ignored_labels, n_classes=num_cls)
    results.append(result)

    mask = np.zeros(label.shape, dtype='bool')
    for l in args.ignored_labels:
        mask[label == l] = True
    
    color_pred_map = convert_to_color(prob_map, palette)
    prob_map[mask] = 0
    mask_color_pred_map = convert_to_color(prob_map, palette)

    file_name = f"{dataset}_num{num}.png"
    model_name = "my_model"
    save_predictions(mask_color_pred_map, color_pred_map, model_name=model_name, caption=file_name)

    show_results(result, label_values=label_values, agregated=False)

    training_time = toc1 - tic1
    testing_time = toc2 - toc1
    print(f"Training takes: {training_time}")
    print(f"Teating takes: {testing_time}")

if num_of_ex > 1:
    show_results(results, label_values=label_values, agregated=True)




