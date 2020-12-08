def plot_decision_boundary(x, y, models, ax, poly_degree=1, test_points=None, shaded=True, interval=np.arange(-6, 6, 0.1)):
    '''
    plot_decision_boundary plots the training data and the decision boundary of the classifier.
    input:
       x - a numpy array of size N x 2, each row is a patient, each column is a biomarker
       y - a numpy array of length N, each entry is either 0 (no cancer) or 1 (cancerous)
       models - an array of classification models
       ax - axis to plot on
       poly_degree - the degree of polynomial features used to fit the model
       test_points - test data
       shaded - whether or not the two sides of the decision boundary are shaded
    returns: 
       ax - the axis with the scatter plot
    
    '''
    # Plot data
    ax.scatter(x[y == 1, 0], x[y == 1, 1], alpha=0.2, c='red', label='class 1')
    ax.scatter(x[y == 0, 0], x[y == 0, 1], alpha=0.2, c='blue', label='class 0')
    
    # Create mesh
    #interval = np.arange(-6, 6, 0.1)
    n = np.size(interval)
    x1, x2 = np.meshgrid(interval, interval)
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    xx = np.concatenate((x1, x2), axis=1)

    # Predict on mesh points
    if(poly_degree > 1):
        polynomial_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
        xx = polynomial_features.fit_transform(xx)
    
    if len(models) > 1:
        alpha_line = 0.1
        linewidths=0.1
    else:
        alpha_line = 0.8
        linewidths=0.5
        
    i = 0
    
    for model in models:
        yy = model.forward(model.weights, xx.T)  
        yy = yy.reshape((n, n))

        # Plot decision surface
        x1 = x1.reshape(n, n)
        x2 = x2.reshape(n, n)
        if shaded:
            ax.contourf(x1, x2, yy, alpha=0.1 * 1. / (i + 1)**2, cmap='bwr',levels=[-0.5,0.5, 1.5])
        CS=ax.contour(x1, x2, yy, colors='black', linewidths=linewidths, alpha=alpha_line,levels=[-0.5,0.5, 1.5])
        # ax.clabel(CS, CS.levels, inline=True, fontsize=10)

        i += 1
        
    if test_points is not None:
        for i in range(len(test_points)):
            pt = test_points[i]
            if i == 0:
                ax.scatter(pt[0], pt[1], alpha=1., s=50, color='black', label='test data')
            else:
                ax.scatter(pt[0], pt[1], alpha=1., s=50, color='black')
        
    ax.set_xlim((-5.5, 5.5))
    ax.set_ylim((-5.5, 5.5))



    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.legend(loc='best')
    return ax

# NEED TO PLOT THE TRAINING DATA SEPARATELY!
def plot_entropycontours(x, y, model, weightlist, ax, title, poly_degree=1, test_points=None, shaded=True, interval=np.arange(-6, 6, 0.1)):
    '''
    plot_decision_boundary plots the training data and the decision boundary of the classifier.
    input:
       x - a numpy array of size N x 2, each row is a patient, each column is a biomarker
       y - a numpy array of length N, each entry is either 0 (no cancer) or 1 (cancerous)
       models - an array of classification models
       ax - axis to plot on
       poly_degree - the degree of polynomial features used to fit the model
       test_points - test data
       shaded - whether or not the two sides of the decision boundary are shaded
    returns: 
       ax - the axis with the scatter plot
    
    '''
    # Plot data
    # ax.scatter(x[y == 1, 0], x[y == 1, 1], alpha=0.2, c='red', label='class 1')
    # ax.scatter(x[y == 0, 0], x[y == 0, 1], alpha=0.2, c='blue', label='class 0')
    
    # Create mesh
    #interval = np.arange(-6, 6, 0.1)
    n = np.size(interval)
    x1, x2 = np.meshgrid(interval, interval)
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    xx = np.concatenate((x1, x2), axis=1)
    print ('xx', xx.shape)
    # Predict on mesh points
    
    yy = myentropy(model, weightlist, xx.T)[1] #change this to entropy func  
    yy = yy.reshape((n, n))
    
    
    cl = plt.imshow(yy, origin='lower', extent=(min(interval), max(interval), min(interval), max(interval)))
    plt.colorbar()
    plt.title(title)
    plt.show()
    
    if test_points is not None:
        for i in range(len(test_points)):
            pt = test_points[i]
            if i == 0:
                ax.scatter(pt[0], pt[1], alpha=1., s=50, color='black', label='test data')
            else:
                ax.scatter(pt[0], pt[1], alpha=1., s=50, color='black')
        
    ax.set_xlim((-5.5, 5.5))
    ax.set_ylim((-5.5, 5.5))
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.legend(loc='best')
    return ax


def myentropy(nn_model, weightlist, xdata):
    '''
    Usage: for NN_Dropout, use the same weights, duplicated N times
    for MFVI, pass the sampled weights
    returns:
    p1narray: Sigmoid probabilities NwxNx
    Hpredcheck: Entropy NwxNx
    '''

    #assert xdata.shape[0]==2
    n_samples = xdata.shape[1]
    p1narray = np.zeros((len(weightlist), n_samples))
    for i, w in enumerate(weightlist):
        p1narray[i, :] = nn_model.forward(w, xdata) #assumes that the 'model.forward' is dropout-like and has generates different outputs for each i
    #print (p_here.shape)
    p2narray = 1 - p1narray
    p1narray = np.mean(p1narray, axis=0)
    p2narray = np.mean(p2narray, axis=0)
    Hpredcheck = -p1narray*np.log(p1narray) - p2narray*np.log(p2narray)
    return p1narray, Hpredcheck
