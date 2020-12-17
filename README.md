# Background
A central problem in Bayesian deep learning (BDL) modelling is uncertainty evaluation, i.e. which models give reasonable epistemic and aleatoric uncertainty estimates. This is a difficult problem to solve because high-dimensional data cannot be visualized in a meaningful way. This problem was encountered early in AM 207. Recall from HW2 that using naive methods for uncertainty evaluation, such as computing the marginal log-likelihood of the data under the posterior, gave misleading results when compared to a visualization of the data and the 95% posterior predictive interval. 

As a result of this difficulty, a set of standardized "benchmarks", or training data, have naturally arisen in the BDL community. One popular benchmark is the UCI datasets, which is a repository of (among other things) classification and regression training data. These benchmarks have well established uncertainty estimates, and in some cases can be meaningfully visualized $^{1}$. These standardized benchmarks are used as training data whenever a new BDL model is proposed. Since the uncertainties are known for these training sets, the proposed model uncertainty estimates can be used as a metric for how well the model fits the data.



# Problem Statement
The paper "A Systematic Comparison of Bayesian Deep Learning Robustness in Diabetic Retinopathy Tasks", which will be reffered to as Filos et al., begins by critiquing the widely used UCI benchmarks:

_"Many BDL papers use benchmarks such as the toy UCI datasets, which consist of only evaluating root mean square error (RMSE) and negative log-likelihood (NLL) on simple datasets with only a few hundred or thousand data points, each with low input and output dimensionality."_

In particular, Filos et al. claims that the UCI datasets are adkin to toy data and do not reflect how real-world data behaves: 

_"Despite BDL’s impact on a range of real-world applications, the development of the field itself is impeded by the lack of realistic benchmarks to guide research... Due to the lack of alternative standard benchmarks, in current BDL research it is common for researchers developing new inference techniques to evaluate their methods with such toy benchmarks alone, ignoring the demands and constraints of the real-world applications which make use of BDL tools"_

Clearly, if the above statements are true, there is a need for a new benchmark which acknowledges the demands and constraints of real-world applications. The paper proposes such a dataset: a training set made up of 512 × 512 RGB images of retinas. The goal is to identify which retinas have a high probability of having diabetic retinopathy and which do not. If the model is uncertain, it should refer the particular picture to a human expert. Filos et al. builds on previous medical work and presents a series of tasks which it claims to be robust against distributional shift and OOD. 


# Existing Work
Previous work on using uncertainty estimates from deep learning models to inform decision referrals in diabetic retinopathy diagnosis was done by Leibig et al. They used the same benchmark as in Filos et al., which is taken from the Kaggle Diabetic Retinopathy (DR) Detection Challenge. Each image is graded by a specialist on the following scale: 0 – No DR, 1 – Mild DR, 2 – Moderate DR, 3 – Severe DR and 4 – Proliferative DR. The 5-class classification task is recast as a binary classification via the following:  sight-threatening DR  is defined as Moderate DR or greater (classes 2-4). The data is augmented using affine transformations, including random zooming (by up to ±10%), random translations (independent shifts by up to ±25 pixels) and random rotations (by up to ±$\pi$). Filos et al. also did this.

Leibig et al. used two deep convolutional neural networks (DCNN) as their architecture and MC dropout to perform approximate inference. One of the DCNN's, named JFnet, is a publicly available network architecture and weights which was trained on  participants who scored very well in the Kaggle DR competition. The other DCNN was built by Leibig et al. and is essentially JFnet with some major tweaks, the largest being that dropout was added after each convolutional layer, hence making this DCNN more Bayesian than JFnet. 

Leibig et al. uses predictive uncertainty, receiver-operating-characteristics (ROC) curves, and the area under a ROC curve, the AUC, as measures of uncertainty for their DCNN's. Importantly, they never claim that their training images nor their methods for uncertainty estimation should be used as a standard benchmark in the BDL community. 

# Contribution 

Filos et al. extends the work done by Leibig et al. Filos et al. fits the diabetic retinopathy diagnosis image data on more BDL methods such as MFVI, deep ensembles, and determinisitc, along with MC dropout. Filos et al. uses the same uncertainty metrics used in Leibig et al. to measure how well the method fit the data. It contrasts these uncertainty estimations with those from the same BDL methods run on UCI training sets. It concludes that these commonly used BDL methods are overfitting their uncertainty in the UCI training data sets (technical details below).

Filos et al then points out that the uncertainty metrics used in Leibig et al. are able to asses robustness to OOD data and distributional shift, and are simple to implement and use**:

_"We extend this methodology (from Leibig et all) with additional tasks that assess robustness to out-of-distribution and distribution shift, using test datasets which were collected using different medical equipment and for different patient populations. Our implementation is easy to use for machine learning researchers who might lack specific domain expertise, since expert details are abstracted away and integrated into metrics which are exposed through a simple API."_ 


Using the arguments above, Filos et al. then proposes the retinopathy diagnosis image data as a standardized benchmark for all new models in the BDL community. Additionally, they propose the uncertainty metrics as a standarized uncertainty metric for the BDL community.


**the validity of this statement is the subject of this group's experimentation


# Technical Details 

Filos et al. used trained all their models on the same images and tackled the same binary classification problem as Leibig et al. (see Existing Work). 


#### Architecture and Training Data


The following information borrows heavily from the "Architecture" section of Filos et al.  Filos et al. used a variety of deep convolutional neural network models, but all are varients of the VGG architecture with 2.5 million parameters, and were all trained using ADAM ($\eta = 4*10^{-4}$, batch size 64). The activation function was a Leaky ReLU for the hidden layers and a sigmoid for the output layer with randomly initialized weights. 


The following information borrows heavily from Section 2 of Filos et al. The Kaggle Diabetic Retinopathy (DR) Detection Challenge data consists of 35,126 training images and 53,576 test images. 20% of the training data is held-out and used as validation. The data is unbalanced: 19.6% of the training set and 19.2% of the test set have a positive label, with positive defined previously (see Existing Work). This inbalance was accounted for by adding more weight to positivly labelled images in the cost/loss function, i.e. the function that ADAM is trying to minimize. 



#### Methods

The four methods used in Filos et al. are MFVI, Deterministic, MC Dropout, and deep ensembles of deterministic and MC Dropout models. All methods were tuned seprately and the results reported in Filos et al. are averages over nine independent models, each using a different random seed. Of these methods, MFVI and determinsitc have already been covered in class. The idea behind deeps ensembles is the same as in HW2: fit a large number of (bootstrap) deep models on the training data, then, at an input  $x$ , use the variance of the ensemble predictions to estimate the uncertainty at  $x$. The remaining method is MC Dropout. 
Dropout is a regularization technique were one randomly zeroes out a set of weights during training. Monte Carlo samples can be drawn from this neural network by using dropout at test time. Finally, for deep ensembles of MC Dropout models: " in this technique, several dropout models are separately trained in parallel. Predictions are then made by sampling repeatedly from all models in the ensemble using fresh dropout masks for each sample, before being averaged, to form a Monte Carlo estimate of the ensemble prediction."

#### Uncertainty Estimation

Fig 4 of Filos et al. shows the relationship between the sigmoid output and the predictive entropy for MC dropout for the correctly and incorreclty labelled images. MC dropout has higher entropy for the miss-classified images! **Filos et al. uses this figure as justification for using predcitive entropy as a measure of uncertainty.** The paper notes that "predictive uncertainty is the sum of epistemic and aleatoric uncertainty", and hence more work is needed to distinguish the two types of uncertanities apart. 

The following information borrows heavily from the "Metrics" section of Filos et al. Recall that the purpose of uncertainty estimation is to flag images where diagnosis in uncertain and refer these images to a medical professional, and relying on the models predictions when it is certain. To simulate this process, uncertainty estimation was measured using "diagnostic accuracy and area under receiver-operating-characteristic (ROC) curve, as a function of the referral rate. We expect the models with well-calibrated uncertainty to refer their least confident predictions to experts ... improving their performance as the number of referrals increases." Diagnostic accuracy is self-explanatory, it is the ratio of the correctly classified data points over the total number of data points. The ROC shows how the diagnostic accuracy changes as a function of referral rate. It is created by plotting the true positive rate, those with  diabetic retinopathy and were correctly labelled as so (sensitivity), vs  false positive rate, i.e. those without  diabetic retinopathy but were labelled as having it (a.k.a. 1 − specificity). The AUC is simply the total area under the curve of the ROC. It varies between 0.5 (chance level) and 1.0 (best possible value). The ROC and AUC plots are included in Figures 5 and 6 of Rilos et al. 



# Experiments


#### ROC and AUC Plots
Let's walk through an example experiment to understand Figure 6, the ROC plots. After training a model, like MC Dropout, on the diabetic retinopathy images, the predictive entropy of each test image is calculated. For a given fraction of retained data, say 0.9, the images with the highest 10% of entropies get thrown out and reffered to a human expert. The remaining 90% of the images are then examined. Each image will have a corresponding sigmoid probability from the model. The threshold for deciding which images have  diabetic retinopathy and which do not based on the sigmoid probabiltiy is varied, resulting on different values for the sensitivity and specificity for each threshold. The plots in Figure 6 are the ROC plots for all the methods at 60% and 90% data retained, using the test images from the Kaggle Diabetic Retinopathy (DR) Detection Challenge data. Since each value for the percentage of data retained yeilds a ROC, it also yeilds a AUC. The plots for AUC vs data retained is in Figure 5a. Figure 5b is simpler: it simply plots the binary accuracy of each method as a function of retained data. No ROC plots were generated. 


#### Robustness to OOD and Distributional Shift

Robustness to distribution shift is evaluated by training on the diabetic retinopathy data detailed in a previous section, and comparing the results to "a completely disjoint APTOS 2019 Blindness Detection dataset collected in India with different medical equipment and on a different population." This is done in Figures 5c and 5d of the paper. The test images are switched from the Kaggle Diabetic Retinopathy (DR) Detection Challenge data to the APTOS Blindness Detection dataset. The general trends stay the same, i.e. Ensemble MC Dropout performs the best out of all the models. These two figures are the justification for claiming that the uncertainty metrics are robust to distributional shift and OOD data. 



# Evaluation 
The paper Filos et al. is overall a well written paper which extends on previous work done by Leibig et al. to build an open-source benchmark with straightforward uncertainty estimates for diagnosing diabetic retinopathy. There is no doubt that the paper will allow for researchers with little medical experience to more easily use state-of-the art BDL algorithms that diagnose diabetic retinopathy. Further claims made by the paper, however, such as claiming that the uncertainty estimates are robust to OOD data and distributional shifts, along with the claim that the diabetic retinopathy benchmark should replace UCI as the standard benchmark for the BDL community, are at best shaky. 

First, the evidence presented by Filos et al. for why their uncertainty estimate metrics are robust to OOD and distributional shift is Fig 5c and 5d. In these figures, all methods were trained using the Kaggle Diabetic Retinopathy (DR) Detection Challenge dataset, and tested on images taken from the APTOS Blindness Detection, which is from India. Though it is true that the APTOS dataset comes from a different country with different medical equipment, medical procedures and equipment tend to be standardized even across different countries. There is no reason to expect medical equipment from the United States to vary in any statistically significant way from those in India. Additionally, both the Kaggle and APTOS images were most likely taken under similar circumstances. It is not clear to the members of this group why the Kaggle and APTOS images must be neccessarily from different distributions or why a large distributional shift is expected among these two datasets. It follows that the members of this group do not believe the conclusion from Filos et al. that the uncertainty estimates are robust to OOD data and distributional shift. 


Second, Filos et al. makes a very strong statement: that the diabetic retinopathy training sets should replace the UCI benchmarks as a standardized benchmark for the BDL community. They claim that the UCI benchmarks are not realistic: 

_"Due to the lack of alternative standard benchmarks, in current BDL research it is common for researchers developing new inference techniques to evaluate their methods with such toy benchmarks alone, **ignoring the demands and constraints of the real-world applications which make use of BDL tools**"_

but the meaning of "the demands and constraints of real-world applications" is never given. What complexities does real-world data have that the UCI benchmark does not? More importantly, why is it about the diabetic retinopathy data that makes it more "real-world" than the UCI data? Is it higher input and output dimensionality? Is it higher class overlap? The fact that Filos et al. never elaborates on this makes the paper's main claim much weaker. 



# Our Experiments
The central contribution of the paper Filos et al. is to use the uncertainty estimation metrics proposed by Leibig et al., as a standardized benchmark for all new models in the BDL community. In particular, the paper claims that classification models such as ensemble methods and MC Dropoup preform better than MFVI on real-world applications (the definition of real world is not made clear). This is somewhat surprising, since, according to reference 4 of Filos et al. ,  MFVI preforms better than these two models on standard UCI benchmarks. 

The paper then concludes that models such as ensemble methods and MC Dropoup are overfitting their uncertainty in the UCI training data sets, which are lower dimensional than the DRD datasets. The purpose of the code below is to explore this claim. In particular, what is it about the diabetic retinopathy data that makes ensemble methods and MC Dropout good fits, and what is it about the UCI data that makes MFVI a good fit? 

We will run simplified versions of the methods developed in Filos et al. on 2D toy data sets with: 

1. Significant class overlap
2. Simulate high dimensional data (i.e. distance between points blown up)
3. Crescent moon 
4. Different shaped Gaussian blobs



To begin with, all our methods (except for MFVI) are neural networks of the following form:
\begin{align}
\mathbf{W} &\sim \mathcal{N}(0, \sigma_{W}^2 \mathbf{I}_{D\times D})\\
\mathbf{w}^{(n)} &= g_{\mathbf{W}}(\mathbf{X}^{(n)})\\
Y^{(n)} &\sim Ber(\text{sigm}(\mathbf{w}^{(n)}))
\end{align}


We modified the neural network and BBVI code from class to implement MFVI, deterministic, MC dropout, and deterministic ensembles. We fine-tuned each model to each toy data set type. We generated data with equal positive and negative classifications to avoid the class imbalance problem that Filos et al. faced. Finally, instead of generating ROC plots, we used the binary accuracy of each model to measure its uncertainty. 
