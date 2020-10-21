import os
import scipy.io as io
from tqdm import tqdm
import numpy as np
import h5py
import random

def read_list(image_path,classes):
    images = []
    cls_id = dict()
    fid = open(classes,'r')
    i=0
    train_list = []
    val_list = []
    for line in fid.readlines():
        data = line.strip('\n').split('	')
        name = data[1]
        
        cls_id[name] = i
        imgDir = image_path + name
        img_list = os.listdir(imgDir)
        
        indexes = np.arange(len(img_list))
        random.shuffle(indexes)
        index = int(round(len(img_list)*0.6))
        trn_idx = indexes[0:index]
        tst_idx = indexes[index:]
        
        for idx in trn_idx:
            item=(img_list[idx],np.int64(i))
            print(item)
            train_list.append(item)
        for idx in tst_idx:
            item=(img_list[idx],np.int64(i))
            print(item)
            val_list.append(item)
        i += 1
        
    fid.close()
    
    return images
    
def make_trans(cls_path):
    images = []
    trans_mat = np.zeros([47,102])
    fine2coarse = dict()
    fid = open(cls_path,'r')
    i = 0
    for line in fid.readlines():
        data = line.strip('\n').split(',')
        for j in data:
            trans_mat[i,int(j)-1] = 1
            fine2coarse[int(j)-1] = i
        i += 1
    fid.close()
    return fine2coarse,trans_mat
    

def checkdir(datapath):
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    return datapath
def checkfile(datapath):
    assert os.path.exists(datapath), 'This is no file %s'%(datapath)
    return datapath
    
            
def save_data():
    print('### Load CUB data')
    print('current path:',os.getcwd())
    ''' path setting '''
    save_data_path = './awa2'
    image_path = '/data/mbobo/Flower/jpg/'
    traindir = os.path.join(image_path,'../train.list')
    valdir = os.path.join(image_path,'../test.list')
    mat_data = io.loadmat('/data/mbobo/Flower/FLO/xian2017/att_splits.mat')
    labels = io.loadmat('imagelabels.mat')['labels'].transpose()
    
    labels = labels.squeeze()-1
    
    
    ''' make c and f data list '''
    fine2coarse,trans_map = make_trans('mapping.txt')
    [nc,nf] = trans_map.shape
    
    
    image_list = os.listdir(image_path)
    
    i=0
    
    train_list = []
    val_list = []
    
    for i in range(102):
    
        idx = np.where(labels==i)
        
        indexes = np.arange(idx[0].shape[0])
        
        random.shuffle(indexes)
        index = int(round(idx[0].shape[0]*0.6))
        trn_idx = indexes[0:index]
        tst_idx = indexes[index:]
        
        for i in trn_idx:
            name = image_list[idx[0][i]]
            label = labels[idx[0][i]]
            item=(name,fine2coarse[label],np.int64(label))
            print(item)
            train_list.append(item)
            
        for i in tst_idx:
            name = image_list[idx[0][i]]
            label = labels[idx[0][i]]
            item=(name,fine2coarse[label],np.int64(label))
            
            val_list.append(item)
            
            
        i += 1
    
    
    ''' att '''
    fine_att = mat_data['att'].transpose()
    coarse_att = np.matmul(trans_map,fine_att)
    coarse_att /= np.sum(trans_map,axis=1,keepdims=True)
    print('coarse_att:',coarse_att.shape)
    print('fine_att:',fine_att.shape)
        
    ''' save '''
    save_path = checkdir(os.path.join('./flo'))
    h5_path = os.path.join(save_path, 'data_info.h5')

    if os.path.exists(h5_path):
        print("Skip store semantic features.")
    else:
        h5_semantic_file = h5py.File(h5_path, 'w')
        # save att
        h5_semantic_file.create_dataset('fine_att', fine_att.shape, dtype=np.float32)
        h5_semantic_file.create_dataset('coarse_att', coarse_att.shape, dtype=np.float32)
        h5_semantic_file.create_dataset('trans_map', trans_map.shape, dtype=np.int16)
        # image path

        h5_semantic_file['fine_att'][...] = fine_att
        h5_semantic_file['coarse_att'][...] = coarse_att
        h5_semantic_file['trans_map'][...] = trans_map
        h5_semantic_file['img_path'] = image_path

        h5_semantic_file.close()

    ''' write visual feats '''
    train_fid = open(save_path+'/train.list','w') 
    test_fid  = open(save_path+'/test.list','w')
    #coarse_fid  = open(save_path+'/coarse_name.list','w')
    
    for item in train_list:
        train_fid.write('{} {} {}\n'.format(item[0],item[1],item[2]))
    train_fid.close()
    
    for item in val_list:
        test_fid.write('{} {} {}\n'.format(item[0],item[1],item[2]))
    test_fid.close()

if __name__ == '__main__':
    save_data()
