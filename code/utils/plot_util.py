import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_3_losses(train_losses,test_losses,TEST_losses):
    plt.figure(figsize = (8,6))
    plt.plot(train_losses, label="training loss in mm")
    plt.plot(test_losses, label="test loss in mm")
    plt.plot(TEST_losses, label="TEST loss in mm")
    plt.legend()

def plot_loss(history):
    """plot training history"""
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

def plot_gt_pre(gt,pre):
    """predictioin vs ground truth in two subplots"""
    plt.figure(figsize = (12,10))
    plt.subplot(211)
    plt.legend()
    ax1=plt.subplot(2, 1, 1)
    ax2=plt.subplot(212, )
    ax1.plot(gt,label = 'ground truth')
    ax1.title.set_text('Ground Truth')
    ax2.plot(pre,label = 'predict')
    ax2.title.set_text('Prediction')

def plot_gt_pre_overlap(gt,pre):
    """predictioin vs ground truth in one plot"""
    fig = plt.figure(figsize =(12,10))
    ax = fig.add_subplot(111)
    ax.plot(gt,label = 'ground truth',zorder=1, color='blue')
    ax.plot(pre, '.', label = 'predict', alpha = 0.5,zorder=3, color='green')
    plt.legend()

# def plot_gt_pre_sep(gt,idx_train,pre_train,idx_test,pre_test):
#     """(predictioin for X_test and X_train separately) vs (ground truth)  in one plot"""
#     fig = plt.figure(figsize =(12,10))
#     ax = fig.add_subplot(111)
    
#     ax.plot(gt,label = 'ground truth',zorder=3, color='blue')
#     ax.plot(idx_train,pre_train,'.', label = 'prediction of training set', alpha = 0.5,zorder=1, color='orange')
#     ax.plot(idx_test,pre_test,'.', label = 'prediction of test set', alpha = 0.5,zorder=2, color='green')
    
    plt.legend()

# def plot_gt_pre_overlap_mul(gt,pre,outputs):
#     """predictioin vs ground truth in one plot
#     for multiple outputs (with names listed in the argument "outputs") in one plot
#     with only first 50000 points"""
#     l = len(outputs)
#     gt = gt.T[:,:50000]
#     pre = pre.T[:,:50000]
#     fig = plt.figure(figsize =(12,8*l))
#     for i,output in enumerate(outputs): 
#         ax = plt.subplot(l,1,i+1)
#         ax.set_title(output, fontsize=16)
#         ax.plot(gt[i],label = 'ground truth',zorder=3, color='blue')
#         ax.plot(pre[i], '.', label = 'predict', alpha = 0.5,zorder=1, color='orange')
#         ax.legend(bbox_to_anchor=(1.25, 1),loc='upper right')


# def plot_gt_pre_sep_mul(gt,pre,idx_train,idx_test,outputs,F=84000):
#     """(predictioin for X_test and X_train separately) vs (ground truth) 
#     for multiple outputs (with names listed in the argument "outputs") in one plot"""
#     l = len(outputs)
#     gt = gt.T
#     pre_train = pre[idx_train].T
#     pre_test = pre[idx_test].T
#     pre = pre.T
#     fig = plt.figure(figsize =(12,8*l))
#     for i,output in enumerate(outputs): 
#         ax = plt.subplot(l,1,i+1)
#         ax.set_title(output, fontsize=16)
#         if 'pos' in output:
#             L = F+10000 # only plot 10000 points for position
#             idx_com_train = np.intersect1d(idx_train,np.arange(F,L))
#             idx_com_test = np.intersect1d(idx_test,np.arange(F,L))
#             ax.plot(np.arange(F,L),gt[i][F:L],label = 'ground truth',zorder=3, color='blue')
#             ax.plot(idx_com_train,pre[i][idx_com_train],'.', label = 'prediction of training set', alpha = 0.5,zorder=1, color='orange')
#             ax.plot(idx_com_test,pre[i][idx_com_test],'.', label = 'prediction of test set', alpha = 0.5,zorder=2, color='green')
#             ax.legend(bbox_to_anchor=(1.25, 1),loc='upper right')
#         else:
#             ax.plot(gt[i],label = 'ground truth',zorder=3, color='blue')
#             ax.plot(idx_train,pre_train[i],'.', label = 'prediction of training set', alpha = 0.5,zorder=1, color='orange')
#             ax.plot(idx_test,pre_test[i],'.', label = 'prediction of test set', alpha = 0.5,zorder=2, color='green')
#             ax.legend(bbox_to_anchor=(1.25, 1),loc='upper right')
        
        

def plot_one_gt_train_test(gt,pred,idx_train,idx_test,F,L,title=""):
    """
    (predictioin for X_test and X_train separately) vs (ground truth) for only one output in one plot.
    Need to define indicies of first point "F", last point "L", and the title in the argument section
    """
    r = np.arange(F,L)
    idx_train = np.intersect1d(idx_train,r)
    idx_test = np.intersect1d(idx_test,r)
    fig,ax = plt.subplots(1,1,figsize =(12,8))
    ax.set_title(title)
    ax.plot(r,gt[r],label = 'ground truth',zorder=3, color='blue')
    ax.plot(idx_train,pred[idx_train],'.', label = 'prediction of training set', alpha = 0.5,zorder=1, color='orange')
    ax.plot(idx_test,pred[idx_test],'.', label = 'prediction of test set', alpha = 0.5,zorder=2, color='green')
    ax.legend(bbox_to_anchor=(1.25, 1),loc='upper right')

def plot_one_gt_train_test_dot_rmse(gt,pred,idx_train,idx_test,F,L,title=""):
    """
    (predictioin for X_test and X_train separately) vs (ground truth) for only one output in one plot.
    print test rmse for this section at the end
    Need to define indicies of first point "F", last point "L", and the title in the argument section
    """

    r = np.arange(F,L)
    idx_train = np.intersect1d(idx_train,r)
    idx_test = np.intersect1d(idx_test,r)
    fig,ax = plt.subplots(1,1,figsize =(12,8))
    ax.set_title(title)
    ax.plot(r,gt[r],'.',label = 'ground truth',zorder=3, color='b')
    ax.plot(idx_train,pred[idx_train],'.', label = 'prediction of training set', alpha = 0.5,zorder=1, color='orange')
    ax.plot(idx_test,pred[idx_test],'.', label = 'prediction of test set', alpha = 0.5,zorder=2, color='green')
    ax.legend(bbox_to_anchor=(1.25, 1),loc='upper right')
    rmse = np.sqrt(np.mean((gt.squeeze()[idx_test]-pred.squeeze()[idx_test])**2))
    print(f"test root_mean_squared_error for this section is {rmse}")