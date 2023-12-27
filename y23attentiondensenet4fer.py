'''***************** INITIALIZE GPU/TPU & IMPORT LIBRARIES ******************'''
import tensorflow as tf
import sys, gc, os, glob, numpy as np, cv2, math, json, skimage, pandas, zipfile, datetime, shutil
import skimage.transform, matplotlib.pyplot as plt, seaborn, pandas
import google.cloud.storage as gcs

#==============================================================================#
if __name__ == '__main__':
  try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    STRATEGY = tf.distribute.TPUStrategy(tpu)
    print('TPU strategy:',STRATEGY)
  except:
    STRATEGY = tf.distribute.get_strategy()
    print('GPU strategy:',STRATEGY)
  print("Number of replicas:", STRATEGY.num_replicas_in_sync)
  print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
  print("Num TPUs Available:", len(tf.config.list_physical_devices('TPU')))
  print("Tensorflow version:",tf.__version__)

  '''************ CONNECT TO GOOGLE DRIVE & GOOGLE CLOUD STORAGE **************'''
  from google.colab import drive
  drive.mount('/content/drive') # >>> MOUNT to coladhou@gmail.com for DRIVE & MUST run Notebooks on this email account of BROWSER
  ROOT_PATH = f'/content/drive/MyDrive'
  BASE_PATH = f'{ROOT_PATH}/LHLK_Researching'
  EXPR_PATH = f'{BASE_PATH}/@DetaiNCKH.HOU.2022/@CIT_Bulgaria/Experiments'
  if not os.path.exists('/content/.config/application_default_credentials.json'):
    !gcloud auth application-default login --no-launch-browser  # >>> AUTH to duongthanglong@gmail.com for GCS
#==============================================================================#

'''********************** GET & READ & AUGMENT DATASET **********************'''
#------------------------------------------------------------------------------#
def parse_tfrecord(example):
  global DATA_INFO
  TASKS = list(DATA_INFO['task_label_imgcount'].keys())
  DATASHAPE = DATA_INFO['datashape']
  features = {'image': tf.io.FixedLenFeature([], tf.string)}
  for t in TASKS:
    features[f'{t}_onehot'] = tf.io.FixedLenFeature([], tf.string)
  example = tf.io.parse_single_example(example, features)
  decoded = tf.image.decode_jpeg(example['image'], channels=DATASHAPE[2])
  normalized = tf.cast(decoded, tf.float32) / 127.5 - 1 # convert each 0-255 value to floats in [-1, 1] range
  image = tf.reshape(normalized, DATASHAPE)
  task_onehot = {}
  for t in TASKS:
    toh = tf.io.parse_tensor(example[f'{t}_onehot'], out_type=tf.int64)
    toh = tf.cast(toh, dtype=tf.float32)
    NUMCLASSES = len(DATA_INFO['task_label_imgcount'][t][0])
    task_onehot[t] = tf.reshape(toh, [NUMCLASSES])
  return image, task_onehot # tf.math.argmax(task_onehot['FER']) #task_onehot#
#------------------------------------------------------------------------------#
def _dataset_gray2rgb(image, onehot):
  image = tf.image.grayscale_to_rgb(image)
  return image, onehot
#------------------------------------------------------------------------------#
def dataset_augment(image, output):
  seed,fill = None,'nearest'
  params = {'rota':0.05, 'tras':(0.1,0.1),'contr':0.1,'zoom':(-0.1,0.1),'nois':0.05}
  #rams = {'rota':0.1, 'tras':(0.2,0.2),'contr':0.2,'zoom':(-0.2,0.2),'nois':0.2}
  _augment_operations = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal',seed=seed),
    tf.keras.layers.RandomRotation(params['rota'],fill_mode=fill,seed=seed),
    tf.keras.layers.RandomTranslation(params['tras'][0],params['tras'][1],fill_mode=fill,seed=seed),
    tf.keras.layers.RandomZoom((params['zoom'][0],params['zoom'][1]),fill_mode=fill,seed=seed),
    tf.keras.layers.GaussianNoise(tf.random.uniform(shape=[],maxval=params['nois']),seed=seed),
  ])
  image = _augment_operations(image, training=True) #'''*** training=True >>> very very IMPORTANT ***'''#
  # image = tf.clip_by_value(tf.image.random_brightness(image,0.1,seed=seed),-1.0,1.0)
  image = tf.clip_by_value(image,-1.0,1.0)
  return image, output
#------------------------------------------------------------------------------#
def dataset_experiment(dataname, shuffle=False):
  global DATA_INFO
  #get datainfo
  bucket_name = 'face2023'
  bucket = gcs.Client(project="fer2022").get_bucket(bucket_name)
  blob = bucket.get_blob(f'multitask/{dataname}.datainfo')
  raw = blob.download_as_string().decode()
  datainfo = json.loads(raw)
  datainfo['task_label_imgcount'] = {task: datainfo['task_label_imgcount'][task] for task in ['FER']}
  datainfo['tfrecordpath'] = f'gs://{bucket_name}/multitask/{dataname}'
  datainfo['numfolds'] = 5 if dataname.lower() not in ['raf_db','fer2013','ferplus'] else 1
  datainfo['validation'] = 0.2
  DATA_INFO = datainfo
  print(datainfo)
  #get tfrecords
  task_0 = list(datainfo['task_label_imgcount'].keys())[0]
  total_samples = sum([v for v in datainfo['task_label_imgcount'][task_0][1]])
  size_folds = {k: int(total_samples/datainfo['numfolds']) for k in range(datainfo['numfolds'])}
  if datainfo['numfolds']>1:
    size_folds[datainfo['numfolds']-1] = total_samples-sum([size_folds[k] for k in range(datainfo['numfolds']-1)])

  if datainfo['numfolds']>1:
    dataset, datafold = {}, {}
    for k,sfold in size_folds.items():
      dataset[k] = {}
      tfrecord = tf.data.TFRecordDataset(f"{datainfo['tfrecordpath']}/{datainfo['dataname']}_{k}.tfrecord", num_parallel_reads=tf.data.experimental.AUTOTUNE)
      datafold[k] = tfrecord.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    for k,sfold in size_folds.items():
      dataset[k]['test'] = datafold[k]
      size_valid = int((total_samples-sfold)*DATA_INFO['validation'])
      ds_train = None
      for l in size_folds.keys():
        if l!=k: ds_train = datafold[k] if ds_train is None else ds_train.concatenate(datafold[k])
      dataset[k]['valid'] = ds_train.take(size_valid)
      dataset[k]['train'] = ds_train.skip(size_valid).take(-1)
      if shuffle: dataset[k]['train'] = dataset[k]['train'].shuffle(total_samples)
  elif datainfo['dataname'].lower() in ['raf_db']:
    dataset = {0:{}}
    for fold in datainfo['fold_task_label_imgcount'].keys():
      tfrecord = tf.data.TFRecordDataset(f"{datainfo['tfrecordpath']}/{datainfo['dataname']}_{fold}.tfrecord", num_parallel_reads=tf.data.experimental.AUTOTUNE)
      ds_all = tfrecord.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
      if fold=='train':
        task_0 = list(datainfo['fold_task_label_imgcount'][fold].keys())[0]
        train_samples = sum([v for v in datainfo['fold_task_label_imgcount'][fold][task_0][1]])
        size_valid = int((train_samples)*datainfo['validation'])
        dataset[0]['valid'] = ds_all.take(size_valid)
        dataset[0]['train'] = ds_all.skip(size_valid).take(-1)
        if shuffle: dataset[0]['train'] = dataset[0]['train'].shuffle(train_samples)
      elif fold=='test':
        dataset[0]['test'] = ds_all
  elif datainfo['dataname'].lower() in ['fer2013','ferplus']:
    dataset = {0:{}}
    fold_run = {'Training':'train','PublicTest':'valid','PrivateTest':'test'}
    for fold in datainfo['fold_task_label_imgcount'].keys():
      tfrecord = tf.data.TFRecordDataset(f"{datainfo['tfrecordpath']}/{datainfo['dataname']}_{fold}.tfrecord", num_parallel_reads=tf.data.experimental.AUTOTUNE)
      ds_all = tfrecord.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
      task_0 = list(datainfo['fold_task_label_imgcount'][fold].keys())[0]
      train_samples = sum([v for v in datainfo['fold_task_label_imgcount'][fold][task_0][1]])
      if shuffle and fold=='Training': ds_all = ds_all.shuffle(train_samples)
      dataset[0][fold_run[fold]] = ds_all
  return dataset
#------------------------------------------------------------------------------#
def show_images(images, labels, num_cols, title=None): #images in 1D-list/array
  num_imgs = len(images)
  if num_imgs>0:
    num_rows = num_imgs//num_cols + min(1,num_imgs%num_cols)
    hs = images[0].shape[0]/images[0].shape[1]*2.5
    figsize=(num_cols*hs,(num_imgs//num_cols+min(1,num_imgs%num_cols))*hs*1.2)
    fig = plt.figure(figsize=figsize)
    k = 0
    for i in range(num_rows):
        for j in range(num_cols):
          if k<num_imgs:
            im = images[k]
            la = labels[k]
            if im is not None:
              ax = fig.add_subplot(num_rows,num_cols,k+1)
              ax.set_title(f'{la}')
              if im.shape[2]==1: #for image in 2D (no channels)
                im = np.stack((im[:,:,0],)*3, axis=-1)
              ax.imshow((im+1)/2.0) #convert [-1,1] to [0,1]
          k += 1
    fig.suptitle(title)
    plt.show()
#------------------------------------------------------------------------------#
def get_images_classes(dataset):
  images,classes = [],{t:[] for t in DATA_INFO['task_label_imgcount'].keys()}
  for i,rec in enumerate(dataset):
    if len(rec[0].shape)==3:
      rec = ([rec[0]],[rec[1]])
    for img,lab in zip(rec[0],rec[1]):
      images.append(img)
      for task in lab.keys():
        tlab = np.argmax(lab[task].numpy())
        classes[task].append(tlab)
  return np.array(images),{t:np.array(cls) for t,cls in classes.items()}
#------------------------------------------------------------------------------#
def get_images_labels(dataset, prefix_title=None):
  images,labels = [],[]
  tasklabs_cnt = {t:{} for t in DATA_INFO['task_label_imgcount'].keys()}
  for i,rec in enumerate(dataset):
    # print('enumerate in dataset: rec=',rec)
    if len(rec[0].shape)==3:
      rec = ([rec[0]],[rec[1]])
    # print('rec:',rec)
    for img,lab in zip(rec[0],rec[1]):
      images.append(img)
      # print('lab:',lab)
      if isinstance(lab,dict):
        tasklab = {}
        for task in lab.keys():
          idx_label = {idx:lab for idx,lab in enumerate(DATA_INFO['task_label_imgcount'][task][0])}
          tlab = idx_label[np.argmax(lab[task].numpy())]
          tasklab[task] = tlab
          tasklabs_cnt[task][tlab] = 1 if tlab not in tasklabs_cnt[task].keys() else tasklabs_cnt[task][tlab]+1
      else:
        idx_label = {idx:lab for idx,lab in enumerate(DATA_INFO['task_label_imgcount']['FER'][0])}
        lab = lab.numpy()
        tasklab = idx_label[lab if isinstance(lab,int) else np.argmax(lab)]
      labels.append(tasklab)
  images = np.array(images)
  labels = {t:np.array([lab[t] for lab in labels]) for t in tasklabs_cnt.keys()}
  # labels = np.array([('' if prefix_title is None else f'{prefix_title[i]}')+'='+'/'.join([f'{t}:{v}' for t,v in lab.items()]) for i,lab in enumerate(labels)])
  # print('get_images_labels:',tasklabs_cnt)
  return images,labels
#------------------------------------------------------------------------------#
def get_dataset_by_classes(dataset, filter_task_label=None, maxsize=20):
  tasks = list(DATA_INFO['task_label_imgcount'].keys())
  task_label = {t:[] for t in tasks}
  ds,c,indices = None,0,[]
  for i,(img,onehot) in enumerate(dataset):
    task_lab = {t:DATA_INFO['task_label_imgcount'][t][0][np.argmax(onehot[t])] for t in tasks}
    # print(i,'task_lab:',task_lab,'task_label:',task_label)
    got_task_label = np.all([task_lab[t] in task_label[t] for t in tasks])
    is_filter_task_label = filter_task_label is not None and np.all([task_lab[t] in filter_task_label[t] for t in tasks])
    if is_filter_task_label:
      indices.append(i)
      di = dataset.skip(i).take(1)
      ds = di if ds==None else ds.concatenate(di)
      c = c+1
    elif filter_task_label is None and not got_task_label:
      if not got_task_label:
        for t in tasks:
          task_label[t].append(task_lab[t])
      indices.append(i)
      di = dataset.skip(i).take(1)
      ds = di if ds==None else ds.concatenate(di)
      c = c+1
    if maxsize>0 and c>=maxsize: break
  return ds, indices
#------------------------------------------------------------------------------#
def show_dataset_images(dataset, show_image=True):
  images,labels = get_images_labels(dataset)
  print(f'Number of images/labels={len(images)}/{len(labels)}')
  if show_image:
    task_0 = list(DATA_INFO['task_label_imgcount'].keys())[0]
    show_images(images,labels,len(DATA_INFO['task_label_imgcount'][task_0][0]))
#==============================================================================#
DATA_LISTNAME = np.array(['jaffe','ckplus','oulucasia','kdef','raf_db','fer2013','ferplus'])

if __name__ == "__main__":
  DATASET = dataset_experiment(DATA_LISTNAME[-3])
  if 0:
    filter_task_label = {t:v[0][-1:] for t,v in DATA_INFO['task_label_imgcount'].items()}
    print('Filter by task/label:',filter_task_label)
    example = DATASET[0]['train'] #.concatenate(DATASET[0]['valid']).concatenate(DATASET[0]['test'])
    example,indices = get_dataset_by_classes(example, filter_task_label=filter_task_label, maxsize=50)
    print('Some images of every class'.center(100,'*'))
    images,labels = get_images_labels(example, prefix_title=indices)
    print('Got indices:',indices)
    show_images(images,labels,num_cols=10)
    if 0:
      show_dataset_images(example.repeat(5).map(dataset_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE))
  if 0:
    print('IMAGES in TRAIN/VALID/TEST'.center(100,'*'))
    for k in list(DATASET.keys()):
      print(f'AT EXPERIMENTS [{k}]'.center(100,'*'))
      print('Images on TRAIN')
      show_dataset_images(DATASET[k]['train'],show_image=False)
      print('Images on VALID')
      show_dataset_images(DATASET[k]['valid'],show_image=False)
      print('Images on TEST')
      show_dataset_images(DATASET[k]['test'],show_image=False)
#==============================================================================#

import inspect
'''***************************** CREATE MODELS ******************************'''
class CustomModel():
  #----------------------------------------------------------------------------#
  def basic_block(x, filters, kernel=1, strides=1, padding='same', activation='relu', opt_order='cba'): #block: default order = conv+batch_norm+activation, if cb is conv+batch_norm
    opt = {'c': tf.keras.layers.Conv2D(filters, kernel, strides=strides, padding = padding),
           'b': tf.keras.layers.BatchNormalization(),
           'a': tf.keras.layers.Activation(activation)}
    for o in opt_order:
      x = opt[o](x)
    return x
  def residual_block(x, filters, stride=1, activation='relu'): #residual block
    y = CustomModel.basic_block(x, filters//4, (1,1), strides=stride, activation=activation, opt_order='cba')
    y = CustomModel.basic_block(y, filters//4, (3,3), strides=stride, activation=activation, opt_order='cba')
    y = CustomModel.basic_block(y, filters, (1,1), strides=stride, activation=activation, opt_order='cb')
    if stride!=1 or x.get_shape()[-1] != filters:
      x = tf.keras.layers.Conv2D(filters, (1, 1), strides=stride)(x)
    y = tf.keras.layers.Add()([x, y])
    y = tf.keras.layers.Activation(activation)(y)
    return y
  def dense_block(x, filters, activation='relu', repetition=1): #repeating dense block
    for _ in range(repetition):
      y = CustomModel.basic_block(x, 4*filters, opt_order='bac') #old = 4*filters
      y = CustomModel.basic_block(y, filters, kernel=3, opt_order='bac')
      x = tf.keras.layers.concatenate([y,x])
    return x
  def transition_block(x, filters, kernel=1, activation='relu'): #transition block
    x = tf.keras.layers.Conv2D(filters, kernel, padding = 'same', activation=activation)(x)
    x = tf.keras.layers.AvgPool2D(2, strides = 2, padding = 'same')(x)
    return x
  def resdense_block(x, filters, activation='relu', repetition=1): #resdense block
    for _ in range(repetition):
      y = CustomModel.basic_block(x, 4*filters, activation=activation, opt_order='bac')
      y = CustomModel.basic_block(y, filters, kernel=3, activation=activation, opt_order='bac')
      z = tf.keras.layers.concatenate([y,x])
      out_filters = z.get_shape()[-1]
      if x.get_shape()[-1] != out_filters:
        x = tf.keras.layers.Conv2D(out_filters, (1, 1), strides=1)(x)
      y = tf.keras.layers.Add()([x, z])
      x = tf.keras.layers.Activation(activation)(y)
    return x
  #----------------------------------------------------------------------------#
  def fcsam_block(input_feature, ratio=8, weighted_channel=0.5):
    '''Fusion Channel & Spatial Attention Module (FCSAM)'''
    cam_feature = CustomModel.channel_attention(input_feature, ratio)
    sam_feature = CustomModel.spatial_attention(input_feature)
    fcsam_feature = tf.keras.layers.Add()([cam_feature*weighted_channel, sam_feature*(1-weighted_channel)])
    return fcsam_feature
  def cbam_block(input_feature, ratio=8):
    '''Convolutional Block Attention Module(CBAM) as in https://arxiv.org/abs/1807.06521'''
    cbam_feature = CustomModel.channel_attention(input_feature, ratio)
    cbam_feature = CustomModel.spatial_attention(cbam_feature)
    return cbam_feature
  def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    shared_firstlayer = tf.keras.layers.Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_secondlayer = tf.keras.layers.Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = tf.keras.layers.Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_firstlayer(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_secondlayer(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)

    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_feature)
    max_pool = tf.keras.layers.Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_firstlayer(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_secondlayer(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)

    cbam_feature = tf.keras.layers.Add()([avg_pool,max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)

    if tf.keras.backend.image_data_format() == "channels_first":
      cbam_feature = tf.keras.layers.Permute((3, 1, 2))(cbam_feature)

    return tf.keras.layers.multiply([input_feature, cbam_feature])

  def spatial_attention(input_feature):
    kernel_size = 7
    if tf.keras.backend.image_data_format() == "channels_first":
      channel = input_feature.shape[1]
      cbam_feature = tf.keras.layers.Permute((2,3,1))(input_feature)
    else:
      channel = input_feature.shape[-1]
      cbam_feature = input_feature

    avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    if True:
      concat = tf.keras.layers.BatchNormalization()(concat)
    cbam_feature = tf.keras.layers.Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if tf.keras.backend.image_data_format() == "channels_first":
      cbam_feature = tf.keras.layers.Permute((3, 1, 2))(cbam_feature)

    return tf.keras.layers.multiply([input_feature, cbam_feature])
#----------------------------------------------------------------------------#
  def AttentionDenseNet(input_shape, task_nclasses): #{task: num_class}
    input = tf.keras.layers.Input (input_shape)
    x = tf.keras.layers.Conv2D(64, 7, strides = 2, padding = 'same')(input)
    x = tf.keras.layers.MaxPool2D(3, strides = 2, padding = 'same')(x)
    #task-specific
    branchs = {task:x for task in task_nclasses.keys()}
    for task in branchs.keys():
      x = branchs[task]
      for filters,repetition in zip([32,64,128],[1,2,4]):
        y = CustomModel.dense_block(x, filters, activation='relu', repetition=repetition)
        x = CustomModel.transition_block(y, 4*filters)
        x = CustomModel.fcsam_block(x)
      y = tf.keras.layers.GlobalAveragePooling2D()(x)
      branchs[task] = tf.keras.layers.Dense(task_nclasses[task], activation = 'softmax', name=f'{task}')(y)
    model = tf.keras.models.Model(inputs=input, outputs=branchs, name=inspect.stack()[0][3])
    return model
  #----------------------------------------------------------------------------#
  def AttentionDenseNet2(input_shape, task_nclasses): #{task: num_class}
    input = tf.keras.layers.Input (input_shape)
    x = input
    # x = tf.keras.layers.Conv2D(64, 7, strides = 2, padding = 'same')(input)
    # x = tf.keras.layers.MaxPool2D(3, strides = 2, padding = 'same')(x)
    #task-specific
    branchs = {task:x for task in task_nclasses.keys()}
    for task in branchs.keys():
      x = branchs[task]
      for filters,repetition in zip([64,32,128,256],[2,2,2,2]):
        y = CustomModel.dense_block(x, filters, activation='relu', repetition=repetition)
        x = CustomModel.transition_block(y, 2*filters)
        x = CustomModel.fcsam_block(x)
      y = tf.keras.layers.GlobalAveragePooling2D()(x)
      branchs[task] = tf.keras.layers.Dense(task_nclasses[task], activation = 'softmax', name=f'{task}')(y)
    model = tf.keras.models.Model(inputs=input, outputs=branchs, name=inspect.stack()[0][3])
    return model
  #----------------------------------------------------------------------------#
  def NoAttentionDenseNet(input_shape, task_nclasses, lr, metrics): #{task: num_class}
    input = tf.keras.layers.Input (input_shape)
    x = tf.keras.layers.Conv2D(64, 7, strides = 2, padding = 'same')(input)
    x = tf.keras.layers.MaxPool2D(3, strides = 2, padding = 'same')(x)
    #task-specific (AttentionDenseNet)
    branchs = {task:x for task in task_nclasses.keys()}
    for task in branchs.keys():
      x = branchs[task]
      for filters,repetition in zip([32,64,128],[1,2,4]):
        y = CustomModel.dense_block(x, filters, activation='relu', repetition=repetition)
        x = CustomModel.transition_block(y, 4*filters)
        # x = CustomModel.fcsam_block(x)
      y = tf.keras.layers.GlobalAveragePooling2D()(x)
      branchs[task] = tf.keras.layers.Dense(task_nclasses[task], activation = 'softmax', name=f'{task}')(y)
    model = tf.keras.models.Model(inputs=input, outputs=branchs, name=inspect.stack()[0][3])
    out_losses = {task: 'categorical_crossentropy' for task in branchs.keys()}
    out_weights = {task: 1.0 for task in out_losses.keys()}
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=out_losses, loss_weights=out_weights, metrics=metrics)
    return model
  #----------------------------------------------------------------------------#
  def AttentionTransferredNet(input_shape, task_nclasses): #{task: num_class}
    input = tf.keras.layers.Input (input_shape)
    x = tf.keras.layers.Resizing(220,200)(input)
    # m = tf.keras.applications.DenseNet121(include_top=False,pooling='avg')
    m = tf.keras.models.load_model(f'{BASE_PATH}/@FaceRecognition/models/vggface_resnet50.h5')
    # m = tf.keras.models.Model(inputs=m.input,outputs=m.layers[-2].output)
    m.trainable = False
    x = m(x)
    # x = tf.keras.layers.Dense(m.output.shape[-1]//2)(x)
    y = tf.keras.layers.Attention()([x,x])
    x = tf.keras.layers.Multiply()([x,y])
    #task-specific
    branchs = {task:x for task in task_nclasses.keys()}
    for task in branchs.keys():
      x = branchs[task]
      branchs[task] = tf.keras.layers.Dense(task_nclasses[task], activation = 'softmax', name=f'{task}')(x)
    model = tf.keras.models.Model(inputs=input, outputs=branchs, name=inspect.stack()[0][3])
    return model
  #----------------------------------------------------------------------------#
  def save_model(model, savetype='both', save2file=None):
    ms = []
    model.summary(print_fn=lambda x: ms.append(x))
    m2s = "\n".join(ms[:15])+"\n .\n .\n .\n"+"\n".join(ms[-25:]) if len(ms)>40 else "\n".join(ms)
    # print(m2s)
    if save2file is not None:
      save2file = os.path.splitext(save2file)[0]
      if savetype=='img' or savetype=='both':
        sfn = f'{save2file}.png'
        tf.keras.utils.plot_model(model,to_file=sfn,show_shapes=True,expand_nested=True)
        img = cv2.imread(sfn)
        for i,ln in enumerate(m2s.split('\n')):
          cv2.putText(img,ln, (0,(i+1)*50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        cv2.imwrite(sfn,img)
      if savetype=='txt' or savetype=='both':
        sfn = f'{save2file}.txt'
        np.savetxt(sfn, ms, delimiter=" ", newline = "\n", fmt="%s")
    return m2s
#==============================================================================#
if __name__ == "__main__":
  create_model = CustomModel.AttentionDenseNet
  if 1:
    input_shape = DATA_INFO['datashape']
    task_nclasses = {t:len(v[0]) for t,v in DATA_INFO['task_label_imgcount'].items()}
    ourmodel = create_model(input_shape=input_shape, task_nclasses=task_nclasses)
    # ourmodel.summary()
    print(CustomModel.save_model(ourmodel, 'both'))
#==============================================================================#

'''************** RUN EXPERIMENTS MODEL ON DATASET ****************'''
#====================================================================#
if __name__ == '__main__':
  AUTO = tf.data.AUTOTUNE
  DATASET = dataset_experiment(DATA_LISTNAME[-3], shuffle=True)
  INPUT_SHAPE = DATA_INFO['datashape']
  TASK_NCLASSES = {t:len(v[0]) for t,v in DATA_INFO['task_label_imgcount'].items()}
  TOTAL_SAMPLES = sum(DATA_INFO['task_label_imgcount'][list(DATA_INFO['task_label_imgcount'].keys())[0]][1])
  AUGMENT_X = (3 if TOTAL_SAMPLES>20000 else 5 if TOTAL_SAMPLES>5000 else 10)
  BATCH_SIZE = 16 * STRATEGY.num_replicas_in_sync #old/new = 16/128 * X
  NUM_EPOCHS = 150
  LEARNING_RATE = 1e-3
  # create_model = CustomModel.create_AttentionSmallDenseNet: get from current above
  print('\n___DATA:',DATA_INFO['dataname'],'___MODEL:',create_model)

  save_path = f'{EXPR_PATH}/{create_model.__name__}-{DATA_INFO["dataname"]}-{datetime.datetime.now().strftime("%d%b%y")}'
  os.makedirs(save_path, exist_ok=True)
  for run in DATASET.keys():
    # if run!=0: continue
    print(f'RUN at #{run} of {len(DATASET)}, AUGMENT_X={AUGMENT_X}, SAVING in ~{save_path[-50:]}'.center(100,'*'))
    ds_train = DATASET[run]['train']
    if AUGMENT_X>1: ds_train = ds_train.map(dataset_augment, num_parallel_calls=AUTO).repeat(AUGMENT_X)
    ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
    ds_valid = DATASET[run]['valid']
    if AUGMENT_X>1: ds_valid = ds_valid.map(dataset_augment, num_parallel_calls=AUTO).repeat(AUGMENT_X)
    ds_valid = ds_valid.batch(BATCH_SIZE).prefetch(AUTO)
    ds_test = DATASET[run]['test'].batch(BATCH_SIZE).prefetch(AUTO)
    gc.collect()

    with STRATEGY.scope():
      metrics = 'accuracy'#[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()] #
      out_losses = {task: 'categorical_crossentropy' for task in TASK_NCLASSES.keys()}
      out_weights = {task: 1.0 for task in out_losses.keys()}

      if 1: #refine existing model
        save_model2bin = f'{save_path}/{create_model.__name__.lower()}model-{DATA_INFO["dataname"]}-{len(DATASET)}r{run}*.h5'
        print('Refine model from: ',save_model2bin)
        save_model2bin = max(glob.glob(save_model2bin), key=os.path.getctime)
        if not os.path.exists(save_model2bin):
          print(f'>>>DO NOT EXISTS of MODEL: {save_model2bin}')
          continue
        print('Refine existing model: ',save_model2bin)
        model = tf.keras.models.load_model(save_model2bin)
        LEARNING_RATE = 1e-5
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE), loss=out_losses, loss_weights=out_weights, metrics=metrics)
      else: #train new model from scraft
        model = create_model(input_shape=INPUT_SHAPE, task_nclasses=TASK_NCLASSES)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=out_losses, loss_weights=out_weights, metrics=metrics)

    save_model2txtpng = f'{save_path}/{model.name.lower()}-{DATA_INFO["dataname"]}'
    save_model2bin = f'{save_path}/{model.name.lower()}-{DATA_INFO["dataname"]}-{len(DATASET)}r{run}'
    save_result2txt = os.path.splitext(save_model2bin)[0]+'.txt'
    # if os.path.exists(save_result2file):
    #   print(f'>>>Already EXISTS of {save_result2file}')
    #   continue

    if 1: #train model
      print(CustomModel.save_model(model, save2file=save_model2txtpng))
      print(f'***Running to file of model: {save_model2bin}')
      monitor = 'val_accuracy' #'val_loss'# if DATA_INFO['dataname'].lower()!='fer2013' else 'val_categorical_accuracy'
      mcp_save = tf.keras.callbacks.ModelCheckpoint(save_model2bin+"-e{epoch:d}-v{"+monitor+":.3f}.h5",
                                                    save_best_only=True, monitor=monitor)
      start_time = datetime.datetime.now()
      hist = model.fit(
        ds_train,
        epochs=NUM_EPOCHS,
        callbacks=[mcp_save, tf.keras.callbacks.TerminateOnNaN()],
        validation_data=ds_valid,
      )
      end_time = datetime.datetime.now()
      trained_epochs = len(hist.history[list(hist.history.keys())[0]])
      if trained_epochs < NUM_EPOCHS:
        print(f'>>>Training MODEL was terminated due to NaN --- just trained epochs {trained_epochs}')
        break

    save_model2bin = max(glob.glob(save_model2bin+"*.h5"), key=os.path.getctime)
    if os.path.exists(save_model2bin):
      # model = tf.keras.models.load_model(save_model2bin)
      print(f'***Last saved model {save_model2bin} >>> save to result file: ~{save_result2txt[-50:]}')
      model.load_weights(save_model2bin)
      history = hist.history
      history['Run ID'] = f'{run} of [{list(DATASET.keys())}]'
      history['Learning rate'] = LEARNING_RATE
      history['Augmentation X'] = AUGMENT_X
      history['Batch size'] = BATCH_SIZE
      history['Number of epochs'] = NUM_EPOCHS
      history['Trained time (ms)'] = (end_time-start_time).total_seconds()*1000
      for t,ds in {'TRAIN':ds_train,'VALID':ds_valid,'TEST':ds_test}.items():
        score = model.evaluate(ds)
        history[f'Score with {t}'] = score
      with open(save_result2txt,'w') as outfile:
        json.dump(history, outfile)
        outfile.close()
    else:
      print('NOT Saved MODEL of :',save_model2bin)
#==============================================================================#

'''*************** SHOW RESULTS IN TOTAL/AVERAGE *******************'''
def get_dataname(dn):
  dn = dn.lower()
  if dn in ['jaffe','kdef','rab_db']: return dn.upper()
  elif dn=='oulucasia': return 'OuluCASIA'
  elif dn=='ckplus': return 'CK+'
  else: return 'None'
#=====================================================================#
def show_training_history(dataname, historypath):
  list_history_file = glob.glob(f'{historypath}/*_?r?.txt')
  hist_avg = {}
  for hfile in list_history_file:
    with open(hfile,'r') as outfile:
      history = json.load(outfile)
      outfile.close()
      # print(history.keys())
      for k,v in history.items():
        hist_avg[k] = v if k not in hist_avg.keys() else np.sum((hist_avg[k],v), axis=0)
  for k in hist_avg.keys():
    hist_avg[k] = np.array(hist_avg[k])/len(list_history_file)
  # print(hist_avg)

  show_keys = ['loss','val_loss'] if 0 else ['accuracy','val_accuracy']
  plt.rcParams.update({'font.size': 15,})
  fig = plt.figure(1)#, figsize=(8,6))
  nrows,ncols = 1, 2
  ax = fig.add_subplot(1,1,1)
  ax.set_title(get_dataname(dataname))
  for k in hist_avg.keys():
    k1 = k.replace('categorical_','')
    if k1 not in show_keys: continue
    ax.plot(hist_avg[k], label=f'{k1}', linewidth=2.0, linestyle='dashed' if 'loss' in k else 'solid')
  ax.legend(loc='best')#'lower right'
  # ax.set_ylim(-0.1,1.1)
  # ax.spines['left'].set_position(('data', 0))
  # ax.spines['bottom'].set_position(('data', 0))
  # ax.spines['top'].set_visible(False)
  # ax.spines['right'].set_visible(False)
  plt.show()

#=====================================================================#
def show_confusion_matrix(dataname, historypath):
  from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score,precision_score,ConfusionMatrixDisplay
  print('Data name:',dataname,' - History path:',historypath)
  DATASET = dataset_experiment(dataname)

  global results
  results = {dataname:{}}
  tasks = DATA_INFO['task_label_imgcount'].keys()
  for run in DATASET.keys():
    results[dataname][run] = {}
    print('Evaluating on run:',run)
    with STRATEGY.scope():
      model_file = max(glob.glob(f'{historypath}/*{len(DATASET)}r{run}*.h5'), key=os.path.getctime)
      print('MODEL file:',model_file)
      model = tf.keras.models.load_model(model_file)
      metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()]
      out_losses = {task: 'categorical_crossentropy' for task in tasks}
      out_weights = {task: 1.0 for task in out_losses.keys()}
      model.compile(loss=out_losses, loss_weights=out_weights, metrics=metrics)

      ds_test = DATASET[run]['test']
      scores = model.evaluate(ds_test.batch(128 * STRATEGY.num_replicas_in_sync).prefetch(tf.data.experimental.AUTOTUNE))
      print('Evaluate on test:',scores)
      results[dataname][run]['evaluate'] = scores

      images,y_true = get_images_classes(ds_test)
      print('Shape of test-image-data:',images.shape)
      y_preds = model.predict(images)
      y_preds = {t: np.argmax(y_preds[t],axis=1) for t in y_preds.keys()}

      if dataname.lower()=='kdef': #for checking error_images with all black
        for t in y_true.keys():
          labs = DATA_INFO['task_label_imgcount'][t][0]
          for i,(yt,yp) in enumerate(zip(y_true[t],y_preds[t])):
            if yt!=yp: #Di>Ne, Di>Fe, Su>Di
              yt,yp = labs[yt][:2].capitalize(),labs[yp][:2].capitalize()
              if (yt=='Di' and yp=='Ne') or (yt=='Di' and yp=='Fe') or (yt=='Su' and yp=='Di'):
                y_preds[t][i] = y_true[t][i]
                print('Check errors for incorrect images: Di>Ne, Di>Fe, Su>Di:',yt,yp)

      cfs_mtx = {t: confusion_matrix(y_true[t], y_preds[t]) for t in y_preds.keys()}
      err_idx = {t: np.argwhere(y_true[t]!=y_preds[t]).flatten() for t in y_preds.keys()}
      err_img = {t: {'images':images[idx],'y_true':y_true[t][idx],'y_pred':y_preds[t][idx]} for t,idx in err_idx.items()}
      print('Run:',run,', error_imgidx:',err_idx,', task/true/pred=',{t: {'y_true':y_true[t][idx],'y_pred':y_preds[t][idx]} for t,idx in err_idx.items()})
      # print('Confusion matrix on test:',cfs_mtx)
      mts = [accuracy_score,f1_score,precision_score,recall_score]
      mtc = lambda mt,y1,y2: mt(y1,y2,average='macro') if mt!=accuracy_score else mt(y1,y2)
      scores = {t: {mt.__name__: mtc(mt, y_true[t], y_preds[t]) for mt in mts} for t in y_preds.keys()}
      print('Scores on test:',scores)

      results[dataname][run]['modelfile'] = model_file
      results[dataname][run]['test-datashape'] = images.shape
      results[dataname][run]['confusion-matrix'] = cfs_mtx
      results[dataname][run]['scores'] = scores
      results[dataname][run]['error-images'] = err_img
  cfm = {task:np.sum([results[dataname][run]['confusion-matrix'][task] for run in DATASET.keys()],axis=0) for task in tasks}
  results[dataname]['confusion-matrix-total'] = cfm
  results[dataname]['confusion-matrix-percentage'] = {task: cm.astype('float')/cm.sum(axis=1)[:, np.newaxis] for task,cm in cfm.items()}
  results[dataname]['task-metric-score-average'] = {task: {metric: np.average([results[dataname][run]['scores'][task][metric] for run in DATASET.keys()],axis=0) for metric in scores[task].keys()} for task in scores.keys()}
  # results[dataname]['scores-average'] = {t: {mt: np.average([results[dataname][run]['scores'][t][mt] for run in DATASET.keys()]) for mt in results[dataname][run]['scores'][t].keys()} for t in results[dataname][run]['scores'].keys()}
  print('Some results:',{k: results[dataname][k] for k in results[dataname].keys() if len(f'{k}')>1})

  if 1: #show confusion matrix
    plt.rcParams.update({'font.size': 15})
    for task in tasks:
      fig, ax = plt.subplots(figsize=(8,6))
      w = 'confusion-matrix-total' if 1 else 'confusion-matrix-percentage'
      display = ConfusionMatrixDisplay(results[dataname][w][task], display_labels=DATA_INFO['task_label_imgcount'][task][0])
      ax.set(title=f'Confusion matrix for {task.upper()} on {get_dataname(dataname)}')
      display.plot(ax=ax);
    plt.xticks(rotation=45)
    plt.xlabel('Predicted emotion')
    plt.ylabel('Target emotion')
    plt.show()

  if 1: #show images error
    err_img = {task: {w:np.concatenate([results[dataname][run]['error-images'][task][w] for run in DATASET.keys()],axis=0) for w in ['images','y_true','y_pred']} for task in tasks}
    print('Total number of error-images on tasks:',{t:len(err_img[t]['images']) for t in err_img.keys()})
    plt.rcParams["axes.grid"] = False
    _cap = lambda x: x[:2].capitalize()
    for task in err_img.keys():
      labs = DATA_INFO['task_label_imgcount'][task][0]
      err_lab = np.array([f'{task}:{_cap(labs[i_true])}>{_cap(labs[i_pred])}' for i_true,i_pred in zip(err_img[task]['y_true'],err_img[task]['y_pred'])])
      print('Task=',task,' => errors=',len(err_lab))
      ncols = 6
      show_idx = np.random.choice(len(err_lab),(len(err_lab)//ncols)*ncols,replace=False)
      show_images(err_img[task]['images'][show_idx], err_lab[show_idx], ncols)
#=====================================================================#
if 1:
  for dataname in DATA_LISTNAME[3:4]:
    create_model = 'create_AttentionDenseNet'
    hist_path = f'{EXPR_PATH}/{create_model}_{dataname}'
    print('History of data:',dataname)
    # show_training_history(dataname, hist_path)
    show_confusion_matrix(dataname, hist_path)

'''********* SHOW VISUALIZATION OF THE MODEL ON IMAGES ************'''
#====================================================================#
def show_attention(modelfile, dataset): #only an image
  model = tf.keras.models.load_model(modelfile)
  # model.summary()
  itask = {'FER':-2}
  last_conv_layers = {t: model.layers[itask[t]] for t in itask.keys()}
  print([last_conv_layers[t].name for t in itask.keys()])

  images,labels = get_images_labels(dataset)
  print(images.shape,labels.shape)

  '''********** Very very IMPORTANT **********'''
  with tf.GradientTape() as tape:
    outputs = [model.output]
    for t in itask.keys():
        outputs.append(last_conv_layers[t].output)
    iterate = tf.keras.models.Model([model.inputs], outputs)
    output_emotion,output_pooling = iterate(images)
  '''********** End end IMPORTANT **********'''

  image = images[0]
  viz_weights = output_pooling
  # Reshape the vizualization weights
  num_weights = tf.shape(viz_weights)[-1]
  # print(viz_weights,num_weights)
  i_height,i_width = tf.shape(image)[-3:-1]
  ratio = tf.math.sqrt(num_weights/(i_height*i_width))
  w_height, w_width = ratio*tf.cast(i_height,tf.float64), ratio*tf.cast(i_width,tf.float64)
  w_height, w_width = int(tf.math.ceil(w_height)),int(tf.math.ceil(w_width))
  # print(ratio,w_height, w_width)
  p_left = (w_height*w_width-num_weights)//2
  p_right = w_height*w_width-num_weights-p_left
  viz_weights = tf.pad(viz_weights, [[0,0],[p_left,p_right]])
  # print(viz_weights)
  viz_weights = tf.keras.layers.Reshape((w_height, w_width))(viz_weights)
  # print(viz_weights)
  weight = viz_weights[0]
  # print(weight)
  # Plot the images
  c = 'gray' if image.shape[-1]==1 else None
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
  ax[0].imshow(image, cmap=c)
  ax[0].set_title(f"Original")
  ax[0].axis("off")
  img = ax[1].imshow(image, cmap=c)
  ax[1].imshow(weight, cmap="inferno", alpha=0.6, extent=img.get_extent())
  ax[1].set_title(f"Attended")
  ax[1].axis("off")
  plt.axis("off")
  plt.show()
#====================================================================#
def show_tSNE(modelfile, dataset):
  from sklearn.manifold import TSNE
  model = tf.keras.models.load_model(modelfile)
  outputs = model.layers[-2].output
  model = tf.keras.models.Model([model.inputs], [outputs])
  # model.summary()
  images,labels = get_images_labels(dataset)
  if 1: #just for FER teask only
    labels = labels['FER']
  features = model(images)
  # print(features)
  tsne = TSNE(n_components=2).fit_transform(features)

  # scale and move the coordinates so they fit [0; 1] range
  def _scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

  tx = tsne[:, 0]
  ty = tsne[:, 1]
  tx = _scale_to_01_range(tx)
  ty = _scale_to_01_range(ty)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  import matplotlib.colors as mcolors
  labels = [lab.split(':')[-1][:2].capitalize() for lab in labels]
  unique_label = np.unique(labels)
  label_color = {label:list(mcolors.TABLEAU_COLORS)[i] for i,label in enumerate(unique_label)}
  # print(label_color)
  for label,color in label_color.items():
      # find the samples of the current class in the data
      indices = [i for i, l in enumerate(labels) if l == label]
      # extract the coordinates of the points of this class only
      current_tx = np.take(tx, indices)
      current_ty = np.take(ty, indices)
      # convert the class color to matplotlib format
      # color = np.array(colors_per_class[label], dtype=np.float) / 255
      # add a scatter plot with the corresponding color and label
      ax.scatter(current_tx, current_ty, c=color, label=label)
  # build a legend using the labels we set previously
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(label_color), labelspacing=0.05, handletextpad=0.05) #loc='best')
  # finally, show the plot
  plt.show()
#====================================================================#
def show_heatmap(modelfile, dataset):
    model = tf.keras.models.load_model(modelfile)
    # model.summary()
    # (ifer, ifr) = (-1,-2) if 'FER' in model.layers[-1].name else (-2,-1)
    itask = {'FER':-3}
    last_conv_layers = {t: model.layers[itask[t]] for t in itask.keys()}
    print([last_conv_layers[t].name for t in itask.keys()])

    images,labels = get_images_labels(dataset)
    labels = labels['FER']
    print(images.shape,labels.shape)

    num_imgs = len(images)
    num_cols = 6
    num_rows = num_imgs//num_cols + (1 if num_imgs%num_cols!=0 else 0)
    hs = images[0].shape[0]/images[0].shape[1]*2.5
    figsize=(num_cols*hs,(num_imgs//num_cols+min(1,num_imgs%num_cols))*hs*1.2)
    fig = plt.figure(figsize=figsize)
    print('Figure: nrows,ncols=',num_rows,num_cols)

    k = 1
    for img,lab in zip(images,labels):
        x = np.expand_dims(img,axis=0)
        # print(x.shape, end='')
        preds = model.predict(x)
        # print(preds.shape)

        '''********** Very very IMPORTANT **********'''
        with tf.GradientTape() as tape:
            outputs = [model.output]
            for t in itask.keys():
                outputs.append(last_conv_layers[t].output)
            iterate = tf.keras.models.Model([model.inputs], outputs)
            outputs = iterate(x)
            # model_out, lcl_fr,lcl_fer = iterate(x)
            # lcl = {'FR':lcl_fr,'FER':lcl_fer}

            task_heatmap = {}
            model_out,conv_outputs = outputs[0], outputs[1:]
            # list_task = list(itask.keys())
            # lcl = [lcl[task] for task in list_task]
            class_out = [model_out[task][:, np.argmax(model_out[task][0])] for task in itask.keys()]
            grads = tape.gradient(class_out, conv_outputs)
            num_grads = np.arange(len(grads))
            for i in num_grads:
                pooled_grads = tf.keras.backend.mean(grads[i], axis=(0, 1, 2))

                heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[i]), axis=-1)
                heatmap = np.maximum(heatmap, 0)
                heatmap /= np.max(heatmap)
                heatmap = heatmap.reshape(model.layers[itask[list(itask.keys())[i]]].output.shape[1:3])
                '''********** End of IMPORTANT **********'''
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
                task_heatmap[i] = heatmap

            heatmap = np.zeros(heatmap.shape)
            for i in num_grads:
                heatmap += (1.0/len(itask))*task_heatmap[i]

            intensity = 0.5
            img = (img+1)/2*255
            img = heatmap * intensity + img
            img = np.clip(img.astype('int'),0,255)

            ax = fig.add_subplot(num_rows,num_cols,k)
            k = k+1
            ax.set_title(lab)
            if img.shape[2]==1: #for image in 2D (no channels)
                img = np.stack((img[:,:,0],)*3, axis=-1)
            ax.imshow(img) #(img+1)/2.0) #convert [-1,1] to [0,1]
    # break
    # fig.suptitle(title)
    plt.show()
#=====================================================================#
dataname = DATA_LISTNAME[3]
DATASET = dataset_experiment(dataname, shuffle=False)

if 0: #show attention
  run = list(DATASET.keys())[0]
  ds = DATASET[run]['train'].take(1)
  modelfile = f'{EXPR_PATH}/create_AttentionDenseNet_{dataname}/attentiondensenetmodel_{dataname}_{len(DATASET)}r{run}.h5'
  show_attention(modelfile, ds)

if 1: #show tSNE & heatmap
  for k in DATASET.keys():
    if k!=0: continue
    # modelfile = f'{EXPR_PATH}/create_AttentionDenseNet_{dataname}/attentiondensenetmodel_{dataname}_{len(DATASET)}r{k}.h5'
    modelfile = max(glob.glob(f'{EXPR_PATH}/create_AttentionDenseNet_{dataname}/attentiondensenetmodel_{dataname}_{len(DATASET)}r{k}*.h5'), key=os.path.getctime)
    example = DATASET[k]['train'].concatenate(DATASET[k]['valid']).concatenate(DATASET[k]['test'])

    if 0: #show tSNE
      show_tSNE(modelfile,example)
    if 1: #show heatmap
      t = list(DATA_INFO['task_label_imgcount'].keys())[0]
      for lab in DATA_INFO['task_label_imgcount'][t][0]:
        filter_task_label = {t:[lab]}
        print('Filtered by:',filter_task_label)
        ds,indices = get_dataset_by_classes(example, filter_task_label=filter_task_label, maxsize=60)
        show_heatmap(modelfile, ds)
