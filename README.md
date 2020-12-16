# Background
A central problem in Bayesian deep learning (BDL) modelling is uncertainty evaluation, i.e. which models give reasonable epistemic and aleatoric uncertainty estimates. This is a difficult problem to solve because high-dimensional data cannot be visualized in a meaningful way. This problem was encountered early in AM 207. Recall from HW2 that using naive methods for uncertainty evaluation, such as computing the marginal log-likelihood of the data under the posterior, gave misleading results when compared to a visualization of the data and the 95% posterior predictive interval. 

As a result of this difficulty, a set of standardized "benchmarks", or training data, have naturally arisen in the BDL community. One popular benchmark is the UCI datasets, which is a repository of (among other things) classification and regression training data. These benchmarks have well established uncertainty estimates, and in some cases can be meaningfully visualized $^{1}$. These standardized benchmarks are used as training data whenever a new BDL model is proposed. Since the uncertainties are known for these training sets, the proposed model uncertainty estimates can be used as a metric for how well the model fits the data.



# Problem Statement
The paper "A Systematic Comparison of Bayesian Deep Learning Robustness in Diabetic Retinopathy Tasks", which will be reffered to as Filos et al, begins by critiquing the widely used UCI benchmarks:

_"Many BDL papers use benchmarks such as the toy UCI datasets, which consist of only evaluating root mean square error (RMSE) and negative log-likelihood (NLL) on simple datasets with only a few hundred or thousand data points, each with low input and output dimensionality."_

In particular, Filos et al claims that the UCI datasets are adkin to toy data and do not reflect how real-world data behaves: 

_"Despite BDL’s impact on a range of real-world applications, the development of the field itself is impeded by the lack of realistic benchmarks to guide research... Due to the lack of alternative standard benchmarks, in current BDL research it is common for researchers developing new inference techniques to evaluate their methods with such toy benchmarks alone, ignoring the demands and constraints of the real-world applications which make use of BDL tools"_

Clearly, if the above statements are true, there is a need for a new benchmark which acknowledges the demands and constraints of real-world applications. The paper proposes such a dataset: a training set made up of 512 × 512 RGB images of retinas. The goal is to identify which retinas have a high probability of having diabetic retinopathy and which do not. If the model is uncertain, it should refer the particular picture to a human expert. Filos et al builds on previous medical work and presents a series of tasks which it claims to be robust against distributional shift and OOD. 


# Existing Work
Previous work on using uncertainty estimates from deep learning models to inform decision referrals in diabetic retinopathy diagnosis was done by Leibig et al. They used the same benchmark as in Filos et al, which is taken from the Kaggle Diabetic Retinopathy (DR) Detection Challenge. Each image is graded by a specialist on the following scale: 0 – No DR, 1 – Mild DR, 2 – Moderate DR, 3 – Severe DR and 4 – Proliferative DR. The 5-class classification task is recast as a binary classification via the following:  sight-threatening DR  is defined as Moderate DR or greater (classes 2-4). The data is augmented using affine transformations, including random zooming (by up to ±10%), random translations (independent shifts by up to ±25 pixels) and random rotations (by up to ±$\pi$). Filos et al also did this.

Leibig et al used two deep convolutional neural networks (DCNN) as their architecture and MC dropout to perform approximate inference. One of the DCNN's, named JFnet, is a publicly available network architecture and weights which was trained on  participants who scored very well in the Kaggle DR competition. The other DCNN was built by Leibig et al and is essentially JFnet with some major tweaks, the largest being that dropout was added after each convolutional layer, hence making this DCNN more Bayesian than JFnet. 

Leibig et al uses predictive uncertainty, receiver-operating-characteristics (ROC) curves, and the area under a ROC curve, the AUC, as measures of uncertainty for their DCNN's. Importantly, they never claim that their training images nor their methods for uncertainty estimation should be used as a standard benchmark in the BDL community. 

# Contribution 

Filos et al extends the work done by Leibig et al. Filos et al fits the diabetic retinopathy diagnosis image data on more BDL methods such as MFVI, deep ensembles, and determinisitc, along with MC dropout. Filos et al uses the same uncertainty metrics used in Leibig et al to measure how well the method fit the data. It contrasts these uncertainty estimations with those from the same BDL methods run on UCI training sets. It concludes that these commonly used BDL methods are overfitting their uncertainty in the UCI training data sets (technical details below).

Filos et al then points out that the uncertainty metrics used in Leibig et al are able to asses robustness to OOD data and distributional shift, and are simple to implement and use**:

_"We extend this methodology (from Leibig et all) with additional tasks that assess robustness to out-of-distribution and distribution shift, using test datasets which were collected using different medical equipment and for different patient populations. Our implementation is easy to use for machine learning researchers who might lack specific domain expertise, since expert details are abstracted away and integrated into metrics which are exposed through a simple API."_ 


Using the arguments above, Filos et al then proposes the retinopathy diagnosis image data as a standardized benchmark for all new models in the BDL community. Additionally, they propose the uncertainty metrics as a standarized uncertainty metric for the BDL community.


**the validity of this statement is the subject of this group's experimentation


# Technical Details 

Filos et al used trained all their models on the same images and tackled the same binary classification problem as Leibig et al (see Existing Work). 


#### Architecture and Training Data


The following information borrows heavily from the "Architecture" section of Filos et al.  Filos et al used a variety of deep convolutional neural network models, but all are varients of the VGG architecture with 2.5 million parameters, and were all trained using ADAM ($\eta = 4*10^{-4}$, batch size 64). The activation function was a Leaky ReLU for the hidden layers and a sigmoid for the output layer with randomly initialized weights. 


The following information borrows heavily from Section 2 of Filos et al. The Kaggle Diabetic Retinopathy (DR) Detection Challenge data consists of 35,126 training images and 53,576 test images. 20% of the training data is held-out and used as validation. The data is unbalanced: 19.6% of the training set and 19.2% of the test set have a positive label, with positive defined previously (see Existing Work). This inbalance was accounted for by adding more weight to positivly labelled images in the cost/loss function, i.e. the function that ADAM is trying to minimize. 





#### Uncertainty Estimation

Fig 4 of Filos et al shows the relationship between the sigmoid output and the predictive entropy for MC dropout for the correctly and incorreclty labelled images. MC dropout has higher entropy for the miss-classified images! **Filos et al uses this figure as justification for using predcitive entropy as a measure of uncertainty.** The paper notes that "predictive uncertainty is the sum of epistemic and aleatoric uncertainty", and hence more work is needed to distinguish the two types of uncertanities apart. 

Recall that the purpose of uncertainty estimation is to flag images where diagnosis in uncertain and refer these images to a medical professional, and relying on the models predictions when it is certain. To simulate this process, uncertainty estimation was measured using "diagnostic accuracy and area under receiver-operating-characteristic (ROC) curve, as a function of the referral rate. We expect the models with well-calibrated uncertainty to refer their least confident predictions to experts ... improving their performance as the number of referrals increases." Diagnostic accuracy is self-explanatory, it is the ratio of the correctly classified data points over the total number of data points. The ROC shows how the diagnostic accuracy changes as a function of referral rate... 


