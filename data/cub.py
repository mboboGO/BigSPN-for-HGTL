import os
import scipy.io as io
import numpy as np
import h5py

def read_list(data_list):
    images = []
    fid = open(data_list,'r')
    for line in fid.readlines():
        data = line.strip('\n').split(' ')
        item=(data[0],np.int64(data[1]))
        images.append(item)
    fid.close()
    return images

def checkdir(datapath):
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    return datapath
def checkfile(datapath):
    assert os.path.exists(datapath), 'This is no file %s'%(datapath)
    return datapath

def att_make(data_list):
    fine2coarse = dict()
    fine_labels = []
    coarse_labels = []
    n = 0
    for item in data_list:
        path = item[0]
        fine_name = path.split('/')[0].split('.')[-1]
        coarse_name = path.split('/')[0].split('_')[-1]
        if coarse_name not in coarse_labels:
            coarse_labels.append(coarse_name)
            n += 1
        if fine_name not in fine_labels:
            fine_labels.append(fine_name)
            fine2coarse[fine_name] = n-1
            
    table = np.zeros([len(coarse_labels),len(fine_labels)])
    for i,label in enumerate(fine_labels):
        j = fine2coarse[label]
        table[j,i] = 1
    
    return coarse_labels,fine_labels,fine2coarse,table

def save_cub_data():
    print('### Load CUB data')
    print('current path:',os.getcwd())
    ''' path setting '''
    split_path = '/data/mbobo/zsl_data//CUB'
    image_path = '/data/mbobo/CUB/CUB_200_2011/images/'
    traindir = os.path.join(image_path,'../train.list')
    valdir = os.path.join(image_path,'../test.list')

    ''' make c and f data list '''
    att_data = io.loadmat(checkfile(os.path.join(split_path, 'att_splits.mat')))
    train_list_ = read_list(traindir)
    val_list_ = read_list(valdir)
    coarse_labels,fine_labels,fine2coarse,trans_map = att_make(train_list_)
    [nc,nf] = trans_map.shape
    
    train_list = []
    for item_ in train_list_:
        path = item_[0]
        name = path.split('/')[0].split('.')[-1]
        item=(path,np.int64(fine2coarse[name]))
        train_list.append(item)
        
    val_list = []
    for item_ in val_list_:
        path = item_[0]
        name = path.split('/')[0].split('.')[-1]
        item=(path,np.int64(fine2coarse[name]),item_[1])
        val_list.append(item)
        
    ''' att '''
    fine_att = att_data['att'].transpose()
    coarse_att = np.matmul(trans_map,fine_att)
    coarse_att /= np.sum(trans_map,axis=1,keepdims=True)
    print('coarse_att:',coarse_att.shape)
    print('fine_att:',fine_att.shape)
        
    ''' save '''
    save_path = checkdir(os.path.join('./cub'))
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
    coarse_fid  = open(save_path+'/coarse_name.list','w')
    
    for item in train_list:
        train_fid.write('{} {} -1\n'.format(item[0],item[1]))
    train_fid.close()
    
    for item in val_list:
        test_fid.write('{} {} {}\n'.format(item[0],item[1],item[2]))
    test_fid.close()
    
    #for item in coarse_name:
    #    coarse_fid.write('{}\n'.format(coarse_name))
    #coarse_fid.close()
    

if __name__ == '__main__':
    save_cub_data()
