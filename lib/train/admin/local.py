
class EnvironmentSettings:
    def __init__(self):

        self.lmdb_dir=""
        
        self.workspace_dir = './'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = './tensorboard/'   # Directory for tensorboard files.
        self.wandb_dir = './wandb/'

        self.lasot_dir = ''
        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''

        self.rgbt210_dir = '/opt/data/VTUAV/RGBT210' 
        self.rgbt234_dir = '/opt/data/VTUAV/RGBT234'
        self.gtot_dir = ''

        self.lasher_dir = "/opt/data/VTUAV/lasher/"
        self.lasher_trainingset_dir = "/opt/data/VTUAV/lasher/train"
        self.lasher_testingset_dir = "/opt/data/VTUAV/lasher/test"
        self.UAV_RGBT_dir = "/opt/data/VTUAV/vtuav_all"