from easydict import EasyDict


cfg = EasyDict()
cfg.data = EasyDict()
cfg.train = EasyDict()
cfg.test = EasyDict()
cfg.network = EasyDict()
cfg.optim = EasyDict()
cfg.checkpoint = './checkpoints'
cfg.codefolder = './hashcodes'
#cfg.task = time.strftime("%Y%m%d-%H%M%S", time.localtime())
cfg.model_name = 'DCMH'
cfg.bit = 16
cfg.continue_train = False

# dataset configure
cfg.data.dataset = 'vggfflickr25'

if cfg.data.dataset == 'vggfflickr25':
    cfg.data.h5_path = '/home/xd/my_project/Conference/NIPS2019/for_lichao/DCMH_PRO/dataset/FLICKR25'
    cfg.data.data_path = '/home/xd/Dataset/flickr25/'
elif cfg.data.dataset == 'nuswide':
    cfg.data.h5_path = '/home/xd/my_project/Conference/NIPS2019/for_lichao/DCMH_PRO/dataset/NUSWIDE'
    cfg.data.data_path = '/home/xd/Dataset/nuswide/'
elif cfg.data.dataset == 'mscoco':
    cfg.data.h5_path = '/home/xd/my_project/Conference/NIPS2019/for_lichao/DCMH_PRO/dataset/MSCOCO'
    cfg.data.data_path = '/home/xd/Dataset/coco2014/'
elif cfg.data.dataset == 'iaprtc12':
    cfg.data.h5_path = '/home/xd/my_project/Conference/NIPS2019/for_lichao/DCMH_PRO/dataset/IAPRTC12'
    cfg.data.data_path = '/home/xd/Dataset/iaprtc12/'
else:
    raise ValueError('No such dataset!')

cfg.data.crop_size = 224


# train configure
cfg.train.batch_size = 128
cfg.train.epochs = 200
cfg.train.lr = 1e-4
cfg.train.lr_step = [30, 60]
cfg.train.gamma = 1
cfg.train.eta = 1

# network configure
cfg.network.pretrain = 'pretrained/VGG/vggf.npy'





