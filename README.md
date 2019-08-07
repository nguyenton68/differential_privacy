# differential_privacy
## Introduction
Differential private is achieve in training process by add noise. Depend on where do you want to add noise, you have different type of perturbation.

**The general training is as follows:**

**Data:** Training data set (X, y)

**Result:** Model parameter \Theta

\Theta <-- Initialize (0)

**#1 Add noise here:** Objective Perturbation: J(\Theta) = loss_function(\Theta, X, y) + Regularization + **Noise**

For epoch in epochs do:
* **#2 Add noise here:** Gradient Perturbation: \Theta = \Theta - learning_rate*(derivative(J(\Theta) + **Noise**)

End


**#3 Add noise here:** Output Perturbation: Return \Theta + **Noise**

## Open source for differential privacy:
* IBM library use objective perturbation to achieve privacy: add noise to the objective function.

    * My study for IBM library can be found here: [IBM DP](https://github.com/Inpher/dp-vs-xor/wiki/IBM-differential-Privacy)
* Tensorflow library use gradient perturbation: add noise to gradient

    * My study for tensorflow library can be found here: [introduction about the library](https://github.com/Inpher/dp-vs-xor/wiki/Tensorflow-with-DP,-note); 
    * Use tensorflow for balanced dataset (MNIST dataset) [slide for MNIST result](https://github.com/Inpher/dp-vs-xor/blob/master/differential_privacy_introduction.pdf);
    * Study result of DP on imbalanced dataset [APS imbalanced dataset](https://github.com/Inpher/dp-vs-xor/wiki/study-DP-with-imbalanced-dataset-(APS-dataset)) and [slide](https://github.com/Inpher/dp-vs-xor/blob/master/DP_with_imbalanced_dataset_v2.pdf)

* There are other methods of achieving DP such as:
  * Input perturbation
  * Sample-aggregate framework
  * Exponential mechanism
  * Teach ensemble framework

